"""
Each data instance from like:
{
  "_id": "652408168c02c8008b71e010",
  "data": [
            {
                "role": "user",
                "parts": [
                    "\u8acb\u4ecb\u7d39\u4f60\u81ea\u5df1\u3002"
                ]
            },
            {
                "role": "model",
                "parts": [
                    "\u60a8\u597d\uff0c\u6211\u662f\u4eba\u5de5\u667a\u80fd\u804a\u5929\u6a5f\u5668\u4eba\u3002\u6211\u662f\u4e00\u500b\u96fb\u8166\u7a0b\u5f0f\uff0c\u88ab\u8a2d\u8a08\u4f86\u56de\u7b54\u60a8\u7684\u554f\u984c\u548c\u8207\u60a8\u4ea4\u8ac7\u3002\u6211\u7684\u77e5\u8b58\u5eab\u975e\u5e38\u5ee3\u6cdb\uff0c\u6db5\u84cb\u5404\u500b\u9818\u57df\uff0c\u5305\u62ec\u79d1\u5b78\u3001\u6b77\u53f2\u3001\u85dd\u8853\u3001\u6587\u5316\u7b49\u3002\u6211\u53ef\u4ee5\u56de\u7b54\u60a8\u7684\u5404\u7a2e\u554f\u984c\uff0c\u5305\u62ec\u65e5\u5e38\u751f\u6d3b\u4e2d\u7684\u554f\u984c\u3001\u77e5\u8b58\u6027\u7684\u554f\u984c\uff0c\u4ee5\u53ca\u4e00\u4e9b\u4f11\u9592\u5a1b\u6a02\u7684\u554f\u984c\u3002\n\n\u6211\u9084\u5728\u4e0d\u65b7\u5b78\u7fd2\u548c\u6210\u9577\u4e2d\uff0c\u5982\u679c\u60a8\u5411\u6211\u63d0\u51fa\u7684\u554f\u984c\uff0c\u6211\u7121\u6cd5\u56de\u7b54\uff0c\u6211\u6703\u53ca\u6642\u5411\u60a8\u5b78\u7fd2\uff0c\u4e26\u5728\u4e0b\u6b21\u7d66\u60a8\u63d0\u4f9b\u6b63\u78ba\u7684\u7b54\u6848\u3002\n\n\u9664\u6b64\u4e4b\u5916\uff0c\u6211\u9084\u5177\u6709\u4e00\u4e9b\u5176\u4ed6\u7684\u529f\u80fd\uff0c\u4f8b\u5982\uff1a\n\n* \u6211\u53ef\u4ee5\u7ffb\u8b6f\u8a9e\u8a00\u3002\n* \u6211\u53ef\u4ee5\u5beb\u8a69\u3002\n* \u6211\u53ef\u4ee5\u5beb\u6545\u4e8b\u3002\n* \u6211\u53ef\u4ee5\u73a9\u904a\u6232\u3002\n\n\u5982\u679c\u60a8\u6709\u4efb\u4f55\u554f\u984c\u6216\u9700\u8981\uff0c\u8acb\u96a8\u6642\u5411\u6211\u63d0\u51fa\uff0c\u6211\u5f88\u6a02\u610f\u70ba\u60a8\u670d\u52d9\u3002"
                ]
            }
  ]
}
to like:
{
    "conversations": [
      {
        "from": "human",
        "value": "用户指令"
      },
      {
        "from": "gpt",
        "value": "模型回答"
      }
    ],
    "system": "",
    "tools": ""
}
"""
import argparse
import json
import json
from datasets import Dataset, DatasetDict, load_dataset


def convert_format(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    new_data = []
    for item in data:
        new_item = {"conversations": [], "system": "", "tools": ""}
        for conversation in item["data"]:
            conv_item = {
                "from": "human" if conversation["role"] == "user" else "gpt",
                "value": "".join(conversation["parts"])
            }
            new_item["conversations"].append(conv_item)
        # assert conversations starts with user and interleaves
        assert len(new_item["conversations"]) % 2 == 0, "對話應該由用戶開始並交錯進行"
        for i in range(0, len(new_item["conversations"]), 2):
            assert new_item["conversations"][i]["from"] == "human", "對話應該由用戶開始"
            assert new_item["conversations"][i+1]["from"] == "gpt", "對話應該由模型回答"
        new_data.append(new_item)

    # 計算並列印一些統計資料
    total_items = len(new_data)
    print(f"總共轉換項目數量: {total_items}")
    data = Dataset.from_list(new_data)
    data.push_to_hub("yentinglin/v1log-geminipro", private=True)

    with open(output_file, 'w') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input file path")
    parser.add_argument("--output", type=str, help="Output file path")
    args = parser.parse_args()

    convert_format(args.input, args.output)
