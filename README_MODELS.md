# 模型文件说明

本项目需要 RF-DETR 预训练模型文件才能运行。由于模型文件体积较大（通常 >100MB），不适合直接放在 Git 仓库中。

## 模型文件存放位置

```
pig_tracking_system/
└── models/
    └── checkpoint_best_ema.pth  # 你的模型文件（名称可自定义）
```

## 如何获取模型

### 选项 1：使用项目提供的模型

如果你是项目团队成员，请通过以下方式获取模型：

1. **联系项目管理员**获取模型下载链接
2. 下载模型文件到本地
3. 将模型文件放到 `models/` 目录

### 选项 2：使用自己训练的模型

如果你有自己训练的 RF-DETR 模型：

1. 将模型文件（`.pth` 格式）复制到 `models/` 目录
2. 运行时指定模型路径：
   ```bash
   python scripts/compare_trackers.py \
       --video_path "input_videos/test.mp4" \
       --model_path "models/your_model.pth"
   ```

### 选项 3：训练自己的模型

参考 [RF-DETR 官方文档](https://github.com/Peterande/RT-DETR) 进行模型训练。

## 模型要求

- **格式**: PyTorch `.pth` 文件
- **架构**: RF-DETR
- **训练目标**: 猪只检测（单类别或多类别）
- **推荐大小**: < 500MB

## 模型配置

模型会自动加载到 GPU（如果可用）。如需修改配置，编辑 `scripts/compare_trackers.py` 中的相关参数。

## 常见问题

### Q: 模型文件太大，如何分享？

建议使用以下方式之一：
- **云存储**: Google Drive, OneDrive, 百度网盘等
- **文件传输服务**: WeTransfer, Send Anywhere等
- **Git LFS**: 如果使用 GitHub，可以考虑 Git Large File Storage

### Q: 如何验证模型是否可用？

运行环境测试脚本：
```bash
python test_environment.py
```

如果显示 `[OK] 找到模型文件: xxx.pth`，说明模型文件正确放置。

### Q: 可以使用 ONNX 格式的模型吗？

当前版本仅支持 PyTorch `.pth` 格式。如需 ONNX 支持，请提交 Issue。

## 模型管理建议

为了方便团队协作：

1. **创建模型说明文档**: 记录模型版本、训练数据、性能指标
2. **使用版本控制**: 为不同版本的模型命名，如 `model_v1.0.pth`
3. **提供下载链接**: 在团队文档中维护最新模型的下载链接
4. **定期备份**: 将重要模型备份到多个位置

## 示例目录结构

```
models/
├── checkpoint_best_ema.pth      # 主模型
├── checkpoint_best_ema_v2.pth   # 备用模型
└── README.txt                    # 模型说明（可选）
```

---

如有其他问题，请查看主 [README.md](README.md) 或提交 Issue。
