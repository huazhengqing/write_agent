import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("Python 路径:")
for i, path in enumerate(sys.path):
    print(f"  {i}: {path}")

print(f"\n项目根目录: {project_root}")
print(f"项目根目录内容: {os.listdir(project_root)}")
print(f"utils 目录存在: {os.path.exists(os.path.join(project_root, 'utils'))}")
if os.path.exists(os.path.join(project_root, 'utils')):
    print(f"utils 目录内容: {os.listdir(os.path.join(project_root, 'utils'))}")

print("\n尝试导入...")

try:
    from utils.rag import RAG
    print("\n✓ 成功导入 utils.rag.RAG")
except Exception as e:
    print(f"\n✗ 导入 utils.rag.RAG 失败: {e}")
    import traceback
    traceback.print_exc()

try:
    from utils.models import Task
    print("✓ 成功导入 utils.models.Task")
except Exception as e:
    print(f"✗ 导入 utils.models.Task 失败: {e}")
    import traceback
    traceback.print_exc()