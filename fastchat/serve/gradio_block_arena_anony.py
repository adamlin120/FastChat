"""
Chatbot Arena (battle) tab.
Users chat with two anonymous models.
"""

import json
import time

import gradio as gr
import numpy as np

from fastchat.constants import (
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    SLOW_MODEL_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
)
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.gradio_block_arena_named import flash_buttons
from fastchat.serve.gradio_web_server import (
    State,
    bot_response,
    get_conv_log_filename,
    no_change_btn,
    enable_btn,
    disable_btn,
    invisible_btn,
    acknowledgment_md,
    ip_expiration_dict,
    get_ip,
    get_model_description_md,
)
from fastchat.utils import (
    build_logger,
    moderation_filter,
)

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

num_sides = 2
enable_moderation = False
anony_names = ["", ""]
models = []


def set_global_vars_anony(enable_moderation_):
    global enable_moderation
    enable_moderation = enable_moderation_


def load_demo_side_by_side_anony(models_, url_params):
    global models
    models = models_

    states = (None,) * num_sides
    selector_updates = (
        gr.Markdown.update(visible=True),
        gr.Markdown.update(visible=True),
    )

    return states + selector_updates


def vote_last_response(states, vote_type, model_selectors, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "models": [x for x in model_selectors],
            "states": [x.dict() for x in states],
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")

    if ":" not in model_selectors[0]:
        for i in range(15):
            names = (
                "### Model A: " + states[0].model_name,
                "### Model B: " + states[1].model_name,
            )
            yield names + ("",) + (disable_btn,) * 4
            time.sleep(0.2)
    else:
        names = (
            "### Model A: " + states[0].model_name,
            "### Model B: " + states[1].model_name,
        )
        yield names + ("",) + (disable_btn,) * 4


def leftvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"leftvote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "leftvote", [model_selector0, model_selector1], request
    ):
        yield x


def rightvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"rightvote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "rightvote", [
            model_selector0, model_selector1], request
    ):
        yield x


def tievote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"tievote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "tievote", [model_selector0, model_selector1], request
    ):
        yield x


def bothbad_vote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"bothbad_vote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "bothbad_vote", [
            model_selector0, model_selector1], request
    ):
        yield x


def regenerate(state0, state1, request: gr.Request):
    logger.info(f"regenerate (anony). ip: {get_ip(request)}")
    states = [state0, state1]
    for i in range(num_sides):
        states[i].conv.update_last_message(None)
    return states + [x.to_gradio_chatbot() for x in states] + [""] + [disable_btn] * 6


def clear_history(request: gr.Request):
    logger.info(f"clear_history (anony). ip: {get_ip(request)}")
    return (
        [None] * num_sides
        + [None] * num_sides
        + anony_names
        + [""]
        + [invisible_btn] * 4
        + [disable_btn] * 2
        + [""]
    )


def share_click(state0, state1, model_selector0, model_selector1, request: gr.Request):
    logger.info(f"share (anony). ip: {get_ip(request)}")
    if state0 is not None and state1 is not None:
        vote_last_response(
            [state0, state1], "share", [model_selector0, model_selector1], request
        )


SAMPLING_WEIGHTS = {
    # zh-tw open-weight models (tier 0)
    "Taiwan-LLM-13B-v2.0-chat": 4,
    "Taiwan-LLM-MoE-alpha": 4,
    "Breeze-7B-Instruct-v0.1": 4,
    # tier 0
    "gpt-4": 4,
    "gpt-4-0314": 4,
    "gpt-4-0613": 4,
    "gpt-4-turbo": 4,
    "gpt-3.5-turbo-0613": 2,
    "gpt-3.5-turbo-1106": 2,
    "claude-2.1": 4,
    "claude-2.0": 2,
    "claude-1": 2,
    "claude-instant-1": 4,
    "gemini-pro": 4,
    "gemini-pro-dev-api": 4,
    "pplx-7b-online": 4,
    "pplx-70b-online": 4,
    "solar-10.7b-instruct-v1.0": 2,
    "llama2-70b-steerlm-chat": 2,
    "mixtral-8x7b-instruct-v0.1": 4,
    "mistral-medium": 4,
    "openhermes-2.5-mistral-7b": 2,
    "dolphin-2.2.1-mistral-7b": 2,
    "wizardlm-70b": 2,
    "starling-lm-7b-alpha": 2,
    "tulu-2-dpo-70b": 2,
    "yi-34b-chat": 2,
    "zephyr-7b-beta": 2,
    "openchat-3.5": 2,
    "chatglm3-6b": 2,
    # tier 1
    "deluxe-chat-v1.2": 4,
    "llama-2-70b-chat": 1.5,
    "llama-2-13b-chat": 1.5,
    "codellama-34b-instruct": 1.5,
    "vicuna-33b": 4,
    "vicuna-13b": 1.5,
    "wizardlm-13b": 1.5,
    "qwen-14b-chat": 1.5,
    "mistral-7b-instruct": 1.5,
    # tier 2
    "vicuna-7b": 1.0,
    "llama-2-7b-chat": 1.0,
    "chatglm2-6b": 1.0,
    # deprecated
    "zephyr-7b-alpha": 1.5,
    "codellama-13b-instruct": 1.0,
    "mpt-30b-chat": 1.5,
    "guanaco-33b": 1.0,
    "fastchat-t5-3b": 0.5,
    "alpaca-13b": 0.5,
    "mpt-7b-chat": 0.1,
    "oasst-pythia-12b": 0.1,
    "RWKV-4-Raven-14B": 0.1,
    "gpt4all-13b-snoozy": 0.1,
    "koala-13b": 0.1,
    "stablelm-tuned-alpha-7b": 0.1,
    "dolly-v2-12b": 0.1,
    "llama-13b": 0.1,
    "chatglm-6b": 0.5,
    "deluxe-chat-v1": 4,
    "palm-2": 1.5,
}

# target model sampling weights will be boosted.
BATTLE_TARGETS = {
    "gpt-4": {"gpt-4-0314", "claude-2.1", "gpt-4-turbo"},
    "gpt-4-0613": {"gpt-4-0314", "claude-2.1", "gpt-4-turbo"},
    "gpt-4-0314": {"gpt-4-turbo", "gpt-4-0613", "claude-2.1", "gpt-3.5-turbo-0613"},
    "gpt-4-turbo": {
        "gpt-4-0613",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-1106",
        "claude-2.1",
        "gemini-pro",
    },
    "gpt-3.5-turbo-0613": {"claude-instant-1", "gpt-4-0613", "claude-2.1"},
    "gpt-3.5-turbo-1106": {"gpt-4-0613", "claude-instant-1", "gpt-3.5-turbo-0613"},
    "solar-10.7b-instruct-v1.0": {
        "mixtral-8x7b-instruct-v0.1",
        "gpt-3.5-turbo-0613",
        "llama-2-70b-chat",
    },
    "mistral-medium": {
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0613",
        "gpt-4-turbo",
        "mixtral-8x7b-instruct-v0.1",
    },
    "mixtral-8x7b-instruct-v0.1": {
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0613",
        "gpt-4-turbo",
        "llama-2-70b-chat",
    },
    "claude-2.1": {"gpt-4-turbo", "gpt-4-0613", "claude-1"},
    "claude-2.0": {"gpt-4-turbo", "gpt-4-0613", "claude-1"},
    "claude-1": {"claude-2.1", "gpt-4-0613", "gpt-3.5-turbo-0613"},
    "claude-instant-1": {"gpt-3.5-turbo-1106", "claude-2.1"},
    "gemini-pro": {"gpt-4-turbo", "gpt-4-0613", "gpt-3.5-turbo-0613"},
    "gemini-pro-dev-api": {"gpt-4-turbo", "gpt-4-0613", "gpt-3.5-turbo-0613"},
    "deluxe-chat-v1.1": {"gpt-4-0613", "gpt-4-turbo"},
    "deluxe-chat-v1.2": {"gpt-4-0613", "gpt-4-turbo"},
    "pplx-7b-online": {"gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "llama-2-70b-chat"},
    "pplx-70b-online": {"gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "llama-2-70b-chat"},
    "openhermes-2.5-mistral-7b": {
        "gpt-3.5-turbo-0613",
        "openchat-3.5",
        "zephyr-7b-beta",
    },
    "dolphin-2.2.1-mistral-7b": {
        "gpt-3.5-turbo-0613",
        "vicuna-33b",
        "starling-lm-7b-alpha",
        "openhermes-2.5-mistral-7b",
    },
    "starling-lm-7b-alpha": {"gpt-3.5-turbo-0613", "openchat-3.5", "zephyr-7b-beta"},
    "tulu-2-dpo-70b": {"gpt-3.5-turbo-0613", "vicuna-33b", "claude-instant-1"},
    "yi-34b-chat": {"gpt-3.5-turbo-0613", "vicuna-33b", "claude-instant-1"},
    "openchat-3.5": {"gpt-3.5-turbo-0613", "llama-2-70b-chat", "zephyr-7b-beta"},
    "chatglm3-6b": {"yi-34b-chat", "qwen-14b-chat"},
    "qwen-14b-chat": {"vicuna-13b", "llama-2-13b-chat", "llama-2-70b-chat"},
    "zephyr-7b-alpha": {"mistral-7b-instruct", "llama-2-13b-chat"},
    "zephyr-7b-beta": {
        "mistral-7b-instruct",
        "llama-2-13b-chat",
        "llama-2-7b-chat",
        "wizardlm-13b",
    },
    "llama-2-70b-chat": {"gpt-3.5-turbo-0613", "vicuna-33b", "claude-instant-1"},
    "llama-2-13b-chat": {"mistral-7b-instruct", "vicuna-13b", "llama-2-70b-chat"},
    "llama-2-7b-chat": {"mistral-7b-instruct", "vicuna-7b", "llama-2-13b-chat"},
    "mistral-7b-instruct": {
        "llama-2-7b-chat",
        "llama-2-13b-chat",
        "llama-2-70b-chat",
    },
    "vicuna-33b": {"llama-2-70b-chat", "gpt-3.5-turbo-0613", "claude-instant-1"},
    "vicuna-13b": {"llama-2-13b-chat", "llama-2-70b-chat"},
    "vicuna-7b": {"llama-2-7b-chat", "mistral-7b-instruct", "llama-2-13b-chat"},
    "wizardlm-70b": {"gpt-3.5-turbo-0613", "vicuna-33b", "claude-instant-1"},
    # zh-tw open-weight models should be sampled together more often
    "Taiwan-LLM-13B-v2.0-chat": {"Taiwan-LLM-MoE-alpha", "Breeze-7B-Instruct-v0.1"},
    "Taiwan-LLM-MoE-alpha": {"Taiwan-LLM-13B-v2.0-chat", "Breeze-7B-Instruct-v0.1", "claude-2.1", "mistral-medium"}
}

SAMPLING_BOOST_MODELS = [
    # "claude-2.1",
    # "gpt-4-0613",
    # "gpt-4-0314",
    # "gpt-4-turbo",
    # "mistral-medium",
    # "llama2-70b-steerlm-chat",
    # "gemini-pro-dev-api",
    "gemini-pro",
    "claude-instant-1.2",
    "mistral-tiny",
    "mistral-small",
]

# outage models won't be sampled.
OUTAGE_MODELS = []


def get_sample_weight(model):
    if model in OUTAGE_MODELS:
        return 0
    weight = SAMPLING_WEIGHTS.get(model, 1.0)
    if model in SAMPLING_BOOST_MODELS:
        weight *= 5
    return weight


def get_battle_pair():
    if len(models) == 1:
        return models[0], models[0]

    model_weights = []
    for model in models:
        weight = get_sample_weight(model)
        model_weights.append(weight)
    total_weight = np.sum(model_weights)
    model_weights = model_weights / total_weight
    chosen_idx = np.random.choice(len(models), p=model_weights)
    chosen_model = models[chosen_idx]
    # for p, w in zip(models, model_weights):
    #     print(p, w)

    rival_models = []
    rival_weights = []
    for model in models:
        if model == chosen_model:
            continue
        weight = get_sample_weight(model)
        if (
            weight != 0
            and chosen_model in BATTLE_TARGETS
            and model in BATTLE_TARGETS[chosen_model]
        ):
            # boost to 50% chance
            weight = total_weight / len(BATTLE_TARGETS[chosen_model])
        rival_models.append(model)
        rival_weights.append(weight)
    # for p, w in zip(rival_models, rival_weights):
    #     print(p, w)
    rival_weights = rival_weights / np.sum(rival_weights)
    rival_idx = np.random.choice(len(rival_models), p=rival_weights)
    rival_model = rival_models[rival_idx]

    swap = np.random.randint(2)
    if swap == 0:
        return chosen_model, rival_model
    else:
        return rival_model, chosen_model


def add_text(
    state0, state1, model_selector0, model_selector1, text, request: gr.Request
):
    ip = get_ip(request)
    logger.info(f"add_text (anony). ip: {ip}. len: {len(text)}")
    states = [state0, state1]
    model_selectors = [model_selector0, model_selector1]

    # Init states if necessary
    if states[0] is None:
        assert states[1] is None

        model_left, model_right = get_battle_pair()
        states = [
            State(model_left),
            State(model_right),
        ]

    if len(text) <= 0:
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [""]
            + [
                no_change_btn,
            ]
            * 6
            + [""]
        )

    model_list = [states[i].model_name for i in range(num_sides)]
    flagged = moderation_filter(text, model_list)
    if flagged:
        logger.info(f"violate moderation (anony). ip: {ip}. text: {text}")
        # overwrite the original text
        text = MODERATION_MSG

    conv = states[0].conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(
            f"conversation turn limit. ip: {get_ip(request)}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [CONVERSATION_LIMIT_MSG]
            + [
                no_change_btn,
            ]
            * 6
            + [""]
        )

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    for i in range(num_sides):
        states[i].conv.append_message(states[i].conv.roles[0], text)
        states[i].conv.append_message(states[i].conv.roles[1], None)
        states[i].skip_next = False

    slow_model_msg = ""
    for i in range(num_sides):
        if "deluxe" in states[i].model_name:
            slow_model_msg = SLOW_MODEL_MSG
    return (
        states
        + [x.to_gradio_chatbot() for x in states]
        + [""]
        + [
            disable_btn,
        ]
        * 6
        + [slow_model_msg]
    )


def bot_response_multi(
    state0,
    state1,
    temperature,
    top_p,
    max_new_tokens,
    request: gr.Request,
):
    logger.info(f"bot_response_multi (anony). ip: {get_ip(request)}")

    if state0 is None or state0.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (
            state0,
            state1,
            state0.to_gradio_chatbot(),
            state1.to_gradio_chatbot(),
        ) + (no_change_btn,) * 6
        return

    states = [state0, state1]
    gen = []
    for i in range(num_sides):
        gen.append(
            bot_response(
                states[i],
                temperature,
                top_p,
                max_new_tokens,
                request,
                apply_rate_limit=False,
            )
        )

    is_gemini = []
    for i in range(num_sides):
        is_gemini.append("gemini" in states[i].model_name)

    chatbots = [None] * num_sides
    iters = 0
    while True:
        stop = True
        iters += 1
        for i in range(num_sides):
            try:
                # yield gemini fewer times as its chunk size is larger
                # otherwise, gemini will stream too fast
                if not is_gemini[i] or (iters % 30 == 1 or iters < 3):
                    ret = next(gen[i])
                    states[i], chatbots[i] = ret[0], ret[1]
                stop = False
            except StopIteration:
                pass
        yield states + chatbots + [disable_btn] * 6
        if stop:
            break


def build_side_by_side_ui_anony(models):
    notice_markdown = """
# ⚔️  繁中 LLM 聊天機器人競技場⚔️ : 野生的大模型測試

## 📜 規則
- 向兩個匿名模型（例如 GPT-4、ChatGPT、Claude、Gemini-Pro、Mistral-Medium、Taiwan-LLM、Breeze）提問，並為較佳者投票！
- 您可以持續對話，直到確定贏家。
- 如果在對話過程中透露了模型身份，則不計入投票。

## 🏆 競技場 Elo 排行榜
我們將在收集到足夠的評分後釋出基於Elo的大型語言模型排行榜，並預計之後每週更新。
找出誰是 🥇繁體中文大型語言模型冠軍！

## 👇 現在就來測試吧！

"""

    states = [gr.State() for _ in range(num_sides)]
    model_selectors = [None] * num_sides
    chatbots = [None] * num_sides

    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Box(elem_id="share-region-anony"):
        with gr.Accordion("🔍 展開以查看競技場選手", open=False):
            model_description_md = get_model_description_md(models)
            gr.Markdown(model_description_md,
                        elem_id="model_description_markdown")
        with gr.Row():
            for i in range(num_sides):
                label = "Model A" if i == 0 else "Model B"
                with gr.Column():
                    chatbots[i] = gr.Chatbot(
                        label=label, elem_id=f"chatbot", height=550
                    )

        with gr.Row():
            for i in range(num_sides):
                with gr.Column():
                    model_selectors[i] = gr.Markdown(anony_names[i])
        with gr.Row():
            slow_warning = gr.Markdown("", elem_id="notice_markdown")

        with gr.Row():
            leftvote_btn = gr.Button(
                value="👈  A表現較佳", visible=False, interactive=False
            )
            rightvote_btn = gr.Button(
                value="👉  B表現較佳", visible=False, interactive=False
            )
            tie_btn = gr.Button(
                value="🤝  平手", visible=False, interactive=False)
            bothbad_btn = gr.Button(
                value="👎  兩者皆差", visible=False, interactive=False
            )

    with gr.Row():
        textbox = gr.Textbox(
            show_label=False,
            placeholder="👉 輸入訊息後按下 ENTER",
            elem_id="input_box",
        )
        send_btn = gr.Button(value="發送", variant="primary", scale=0)

    with gr.Row() as button_row:
        clear_btn = gr.Button(value="🎲 新一輪", interactive=False)
        regenerate_btn = gr.Button(value="🔄  重新生成", interactive=False)
        share_btn = gr.Button(value="📷  分享")

    with gr.Accordion("生成參數", open=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="隨機性",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P (機率閾值)",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=2048,
            value=1024,
            step=64,
            interactive=True,
            label="最大輸出字數",
        )

    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    # Register listeners
    btn_list = [
        leftvote_btn,
        rightvote_btn,
        tie_btn,
        bothbad_btn,
        regenerate_btn,
        clear_btn,
    ]
    leftvote_btn.click(
        leftvote_last_response,
        states + model_selectors,
        model_selectors + [textbox, leftvote_btn,
                           rightvote_btn, tie_btn, bothbad_btn],
    )
    rightvote_btn.click(
        rightvote_last_response,
        states + model_selectors,
        model_selectors + [textbox, leftvote_btn,
                           rightvote_btn, tie_btn, bothbad_btn],
    )
    tie_btn.click(
        tievote_last_response,
        states + model_selectors,
        model_selectors + [textbox, leftvote_btn,
                           rightvote_btn, tie_btn, bothbad_btn],
    )
    bothbad_btn.click(
        bothbad_vote_last_response,
        states + model_selectors,
        model_selectors + [textbox, leftvote_btn,
                           rightvote_btn, tie_btn, bothbad_btn],
    )
    regenerate_btn.click(
        regenerate, states, states + chatbots + [textbox] + btn_list
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )
    clear_btn.click(
        clear_history,
        None,
        states + chatbots + model_selectors +
        [textbox] + btn_list + [slow_warning],
    )

    share_js = """
function (a, b, c, d) {
    const captureElement = document.querySelector('#share-region-anony');
    html2canvas(captureElement)
        .then(canvas => {
            canvas.style.display = 'none'
            document.body.appendChild(canvas)
            return canvas
        })
        .then(canvas => {
            const image = canvas.toDataURL('image/png')
            const a = document.createElement('a')
            a.setAttribute('download', 'chatbot-arena.png')
            a.setAttribute('href', image)
            a.click()
            canvas.remove()
        });
    return [a, b, c, d];
}
"""
    share_btn.click(share_click, states + model_selectors, [], _js=share_js)

    textbox.submit(
        add_text,
        states + model_selectors + [textbox],
        states + chatbots + [textbox] + btn_list + [slow_warning],
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons,
        [],
        btn_list,
    )

    send_btn.click(
        add_text,
        states + model_selectors + [textbox],
        states + chatbots + [textbox] + btn_list,
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )

    return states + model_selectors
