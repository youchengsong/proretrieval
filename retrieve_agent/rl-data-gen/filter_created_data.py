import asyncio
import json
import traceback

from retrieve_env import execute_sql


async def main():
    data = []
    with open('/workspace/yunhai/rl-date-gen/email_query.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    res = []
    for item in data:
    
        try:
            print(item)
            # 清洗 sql 字段
                # 解析 label 字段（Python 字典格式字符串，使用 ast.literal_eval）
            sql1 = item['sql'].replace("DISTINCT ", "").replace(';','').replace('SELECT message_id','SELECT *')
            sql1 = {
                "sql": sql1,
                "vector_query_list": item.get('vector_query_list', []),
            }
            
            print("sql1"+str(sql1))

            res1 = execute_sql(str(sql1), 50)
            print(res1)
            
            if not res1:
                print("❌ SQL 执行失败，跳过此条记录")
                continue
            elif not res1['result']:
                print("❌ SQL 执行失败，跳过此条记录")
                continue
            else:
                res1 = [ it['id'] for it in res1['result']]
            if item['message_id'] in res1:
                acc = 1.0
            else:
                acc = 0.0   

            if acc == 1.0:
                res.append(item)
            # 转换为标准 JSON 字符串

        except Exception as e:
            print("Error processing item:", item)
            traceback.print_exc()
            continue

    with open("/workspace/yunhai/rl-date-gen/email_query_with_sql2.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())