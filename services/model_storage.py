
"""
Cloudflare R2 模型存储管理
支持：
1. 上传 models/ 目录中的模型文件
2. 从 R2 下载并恢复模型
3. Windows 本地开发和 Render 部署
4. 自动读取项目根目录 .env
5. 分页列举、临时文件下载、原子替换
首次上传模型
python -m services.model_storage upload
测试 R2 连接
python -m services.model_storage test
"""

import os
import logging
from pathlib import Path
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv


# ============================================================
# 1. 基础目录配置
# ============================================================

# 当前文件位于 services/model_storage.py
# parent.parent 即项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 模型目录
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 加载项目根目录 .env
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=False)


# ============================================================
# 2. R2 配置
# ============================================================

R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "").strip()
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "").strip()
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "").strip()

# R2 对象前缀
R2_PREFIX = os.getenv("R2_PREFIX", "python-ml").strip().strip("/")

# 是否强制要求 R2 配置成功
R2_REQUIRED = os.getenv("R2_REQUIRED", "false").strip().lower() in (
    "1", "true", "yes", "on"
)

# 可选自定义 Endpoint
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "").strip()

# 日志
logger = logging.getLogger(__name__)


# ============================================================
# 3. 模型文件识别
# ============================================================

MODEL_EXTENSIONS = {
    ".h5",
    ".pkl",
    ".keras",
    ".joblib",
}


def is_model_file(path: Path) -> bool:
    """
    判断是否为需要上传的模型文件。

    支持：
    .h5
    .pkl
    .keras
    .joblib
    *_feature_weights.json
    """

    if not path.is_file():
        return False

    if path.suffix.lower() in MODEL_EXTENSIONS:
        return True

    if path.name.lower().endswith("_feature_weights.json"):
        return True

    return False


# ============================================================
# 4. 检查配置
# ============================================================

def check_r2_config() -> bool:
    """检查 R2 配置是否完整。"""

    required = {
        "R2_ACCOUNT_ID": R2_ACCOUNT_ID,
        "R2_ACCESS_KEY_ID": R2_ACCESS_KEY_ID,
        "R2_SECRET_ACCESS_KEY": R2_SECRET_ACCESS_KEY,
        "R2_BUCKET_NAME": R2_BUCKET_NAME,
    }

    missing = [
        name for name, value in required.items()
        if not value
    ]

    if missing:
        message = (
            "[R2] R2 配置不完整："
            + ", ".join(missing)
        )

        if R2_REQUIRED:
            raise RuntimeError(message)

        logger.warning(message)
        return False

    return True


# ============================================================
# 5. 创建 R2 S3 客户端
# ============================================================

def get_s3_client():
    """创建 Cloudflare R2 S3 兼容客户端。"""

    if not check_r2_config():
        return None

    endpoint = R2_ENDPOINT_URL or (
        f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    )

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            retries={
                "max_attempts": 3,
                "mode": "standard",
            },
            connect_timeout=10,
            read_timeout=60,
        ),
    )

    return client


# ============================================================
# 6. R2 对象路径
# ============================================================

def get_object_prefix() -> str:
    """
    R2 对象统一存放位置。

    例如：
    python-ml/models/BTCUSDT_enhanced_lstm.h5
    """

    if R2_PREFIX:
        return f"{R2_PREFIX}/models/"

    return "models/"


def get_object_key(local_file: Path) -> str:
    """将本地模型路径转换为 R2 对象路径。"""

    relative_path = local_file.relative_to(MODELS_DIR)

    return (
        get_object_prefix()
        + relative_path.as_posix()
    )


# ============================================================
# 7. 扫描本地模型
# ============================================================

def scan_local_models():
    """递归扫描 models/ 中的模型文件。"""

    if not MODELS_DIR.exists():
        return []

    files = []

    for path in MODELS_DIR.rglob("*"):
        if is_model_file(path):
            files.append(path)

    return sorted(files)


# ============================================================
# 8. 上传模型到 R2
# ============================================================

def upload_models() -> int:
    """
    上传本地全部模型文件到 R2。

    返回成功上传的文件数量。
    """

    print("[R2] 正在准备上传模型...")

    client = get_s3_client()

    if client is None:
        print("[R2] 配置不完整，跳过模型上传")
        return 0

    files = scan_local_models()

    if not files:
        print("[R2] 没有找到需要上传的模型文件")
        return 0

    uploaded = 0

    for local_file in files:
        object_key = get_object_key(local_file)

        try:
            client.upload_file(
                str(local_file),
                R2_BUCKET_NAME,
                object_key,
            )

            uploaded += 1

            print(
                f"[R2] 上传成功：{local_file.relative_to(MODELS_DIR)}"
            )

        except Exception as exc:
            logger.exception(
                "[R2] 上传失败：%s",
                local_file
            )

            print(f"[R2] 上传失败：{local_file.name}，原因：{exc}")

    print(
        f"[R2] 模型上传完成：{uploaded}/{len(files)} 个文件"
    )

    return uploaded


# ============================================================
# 9. 下载模型到本地
# ============================================================

def download_models() -> int:
    """
    从 R2 下载全部模型文件到 models/。

    使用分页列举对象。
    下载到 .part 临时文件，成功后再替换正式文件。

    返回成功下载的文件数量。
    """

    print("[R2] 正在检查 Cloudflare R2 模型存储...")

    client = get_s3_client()

    if client is None:
        print("[R2] 未配置完整，跳过云端模型恢复")
        return 0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    prefix = get_object_prefix()

    downloaded = 0
    found = 0

    paginator = client.get_paginator("list_objects_v2")

    try:
        pages = paginator.paginate(
            Bucket=R2_BUCKET_NAME,
            Prefix=prefix,
        )

        for page in pages:
            objects = page.get("Contents", [])

            for obj in objects:
                object_key = obj.get("Key", "")

                if not object_key or object_key.endswith("/"):
                    continue

                # 去掉 R2 前缀，得到模型相对路径
                relative_key = object_key[len(prefix):]

                if not relative_key:
                    continue

                # 只下载支持的模型文件
                relative_path = Path(relative_key)

                if not is_model_file_name(relative_path.name):
                    continue

                found += 1

                # 防止异常路径写出 models 目录
                local_file = (MODELS_DIR / relative_path).resolve()

                try:
                    local_file.relative_to(MODELS_DIR.resolve())
                except ValueError:
                    logger.warning(
                        "[R2] 忽略异常对象路径：%s",
                        object_key
                    )
                    continue

                local_file.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                # 临时文件
                temp_file = local_file.with_name(
                    local_file.name + ".part"
                )

                try:
                    client.download_file(
                        R2_BUCKET_NAME,
                        object_key,
                        str(temp_file),
                    )

                    # 确认下载文件大小
                    expected_size = obj.get("Size")

                    if (
                        expected_size is not None
                        and temp_file.stat().st_size != expected_size
                    ):
                        raise IOError(
                            f"文件大小不一致："
                            f"预期 {expected_size}，"
                            f"实际 {temp_file.stat().st_size}"
                        )

                    # 下载成功后替换正式文件
                    os.replace(temp_file, local_file)

                    downloaded += 1

                    print(
                        f"[R2] 下载成功：{relative_key}"
                    )

                except Exception:
                    if temp_file.exists():
                        temp_file.unlink()

                    raise

        print(
            f"[R2] 模型恢复完成：{downloaded} 个文件"
        )

        if found == 0:
            message = (
                f"[R2] Bucket 中没有找到模型文件，"
                f"检查对象前缀：{prefix}"
            )

            if R2_REQUIRED:
                raise RuntimeError(message)

            print(message)

        elif downloaded == 0 and R2_REQUIRED:
            raise RuntimeError(
                "[R2] 找到了模型对象，但没有成功下载任何文件"
            )

        return downloaded

    except Exception as exc:
        logger.exception("[R2] 模型下载失败")

        print(f"[R2] 模型恢复失败：{exc}")

        if R2_REQUIRED:
            raise

        return downloaded


def is_model_file_name(filename: str) -> bool:
    """根据文件名判断是否为模型文件。"""

    name = filename.lower()
    suffix = Path(name).suffix

    if suffix in MODEL_EXTENSIONS:
        return True

    return name.endswith("_feature_weights.json")


# ============================================================
# 10. 测试 R2 连接
# ============================================================

def test_r2_connection() -> bool:
    """测试 R2 Bucket 是否可以访问。"""

    print("[R2] 正在测试连接...")

    client = get_s3_client()

    if client is None:
        return False

    try:
        client.head_bucket(
            Bucket=R2_BUCKET_NAME
        )

        print("[R2] 连接成功")
        print(f"[R2] Bucket：{R2_BUCKET_NAME}")
        print(f"[R2] Prefix：{get_object_prefix()}")

        return True

    except ClientError as exc:
        logger.exception("[R2] Bucket 访问失败")
        print(f"[R2] 连接失败：{exc}")
        return False

    except Exception as exc:
        logger.exception("[R2] 连接异常")
        print(f"[R2] 连接异常：{exc}")
        return False


# ============================================================
# 11. 命令行入口
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Cloudflare R2 模型存储管理工具"
    )

    parser.add_argument(
        "action",
        choices=[
            "upload",
            "download",
            "test",
            "list",
        ],
        help="操作类型",
    )

    args = parser.parse_args()

    if args.action == "upload":
        upload_models()

    elif args.action == "download":
        download_models()

    elif args.action == "test":
        test_r2_connection()

    elif args.action == "list":
        files = scan_local_models()

        print(f"本地模型目录：{MODELS_DIR}")
        print(f"找到 {len(files)} 个模型文件：")

        for file in files:
            print(
                f"  {file.relative_to(MODELS_DIR)}"
                f" ({file.stat().st_size} bytes)"
            )
