import asyncio
import json
import traceback

from retrieve_env import execute_sql, search_mail
from dotenv import load_dotenv
from openai import OpenAI



async def main():
    data = []
    with open('/workspace/yunhai/rl-date-gen/pipeline_results_v3.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    res = []
    empty = []
    error = []
    with open("/workspace/yunhai/rl-date-gen/passed_result.jsonl", 'w', encoding='utf-8') as f_passed, \
            open("/workspace/yunhai/rl-date-gen/empty_result.jsonl", 'w', encoding='utf-8') as f_empty, \
            open("/workspace/yunhai/rl-date-gen/error_result.jsonl", 'w', encoding='utf-8') as f_error:
        for item in data:
            try:
                print(item)
                query = item.get('raw_query', None)
                if not query:
                    print("❌ 查询语句为空，跳过此条记录")
                    continue
                
                res1 = search_mail(str(query), 10)
                print(res1)
                item['stmt'] = res1.get('stmt', None)
                item['monc_sql'] = res1.get('monc_sql', None)
                if not res1:
                    print("❌ Query 执行失败，跳过此条记录")
                    error.append(item)
                    line = json.dumps(item, ensure_ascii=False)
                    f_error.write(line + '\n')
                    continue
                elif not res1['result']:
                    print("❌ Query 没有找到任何结果")
                    empty.append(item)
                    line = json.dumps(item, ensure_ascii=False)
                    f_empty.write(line + '\n')
                    continue
                else:
                    res1 = [ it['id'] for it in res1['result']]
                if item['gold_id'] in res1:
                    acc = 1.0
                    line = json.dumps(item, ensure_ascii=False)
                    f_passed.write(line + '\n')
                else:
                    acc = 0.0   
                    line = json.dumps(item, ensure_ascii=False)
                    f_empty.write(line + '\n')

            except Exception as e:
                print("Error processing item:", item)
                traceback.print_exc()
                continue

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())