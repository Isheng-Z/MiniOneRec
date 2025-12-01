import os
from huggingface_hub import snapshot_download

# 配置下载路径
download_path = os.path.join(os.getcwd(), "models")

# 如果你在国内且没有开全局代理，取消下面这行的注释使用镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("开始从 Hugging Face 下载 Qwen2.5-0.5B...")

model_dir = snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B",
    local_dir=os.path.join(download_path, "Qwen2.5-0.5B"),
    local_dir_use_symlinks=False,  # 确保下载的是实体文件而不是软链接
    resume_download=True
)

print(f"下载完成！模型路径为: {model_dir}")