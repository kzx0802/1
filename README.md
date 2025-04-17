import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn
import torch.nn.utils.prune as prune
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models.detection import COCOEvaluator
from torchvision.models.detection import GeneralizedRCNNTransform
from torch.utils.data import Dataset
import os

# ========== 配置参数 ==========
COCO_IMG_DIR = "C:/Users/Zhenxi.Kong23/Desktop/New folder/coco/val2017"
COCO_ANN_FILE = "C:/Users/Zhenxi.Kong23/Desktop/New folder/coco/annotations/instances_val2017.json"
PRUNE_RATIOS = [0.0, 0.01, 0.036, 0.062, 0.087, 0.113, 0.139, 0.165, 0.191, 0.216, 0.242, 0.268, 0.294, 0.319, 0.345, 0.371, 0.397, 0.423, 0.448, 0.474, 0.5]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== 数据准备 ==========
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.CocoDetection(root=COCO_IMG_DIR, annFile=COCO_ANN_FILE, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# ========== 剪枝函数 ==========
def prune_model(model, amount):
    prune_report = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune_report[name] = {
                "prune_ratio": amount,
                "remaining_weights": int(torch.sum(module.weight != 0))
            }
    return prune_report

# ========== COCO mAP 计算器（torchvision） ==========
def evaluate_model_map(model, data_loader, max_samples=100):
    model.eval()
    coco_results = []
    count = 0

    # COCOEvaluator 用于评估
    coco_evaluator = COCOEvaluator(data_loader.dataset)

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            # 直接把结果传给 COCOEvaluator
            coco_evaluator.update(outputs, targets)
            count += 1
            if count >= max_samples:
                break

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return coco_evaluator.coco_eval['bbox'].stats[0]  # 返回 mAP@[0.5:0.95]

# ========== 主流程 ==========
results = []

for ratio in PRUNE_RATIOS:
    print(f"\n>>> 正在剪枝比例: {ratio}")
    model = retinanet_resnet50_fpn(pretrained=True)
    model.to(DEVICE)
    model.eval()

    if ratio > 0:
        report = prune_model(model, ratio)
        print(f"=== Pruning Report (Ratio={ratio}) ===")
        for layer, info in report.items():
            print(f"Layer: {layer} | Pruned: {ratio*100:.2f}% | Remaining: {info['remaining_weights']} weights")

    map_value = evaluate_model_map(model, data_loader)
    print(f"[*] 剪枝比例 {ratio:.3f} -> mAP: {map_value:.4f}")
    results.append((ratio, map_value))

# ========== 打印总结 ==========
print("\n========== RetinaNet 剪枝评估结果（mAP） ==========")
print("剪枝比例\tmAP")
for r, m in results:
    print(f"{r:.3f}\t\t{m:.4f}")
