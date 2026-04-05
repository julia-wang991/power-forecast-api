# Power Forecast API for Lovable

这是基于你现有 Colab 预测逻辑整理好的 FastAPI 部署版本，已经包含：

- `GET /dashboard`：直接给 Lovable 首页卡片和 24 小时图表用
- `GET /forecast?hours=24`：返回逐小时预测明细
- `GET /daily-summary?hours=168`：返回按天汇总结果
- `GET /health`：健康检查

## 目录说明

- `app.py`：主程序
- `requirements.txt`：依赖
- `render.yaml`：Render 部署配置
- `load_shape_model.pkl`
- `trend_df.pkl`
- `climatology.pkl`
- `meta.pkl`

## 本地运行

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

打开：

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/dashboard`
- `http://127.0.0.1:8000/forecast?hours=24`
- `http://127.0.0.1:8000/daily-summary?hours=168`

## 部署到 Render

1. 新建 GitHub 仓库
2. 把整个目录上传到仓库根目录
3. 登录 Render
4. 选择 **New +** → **Web Service**
5. 连接你的 GitHub 仓库
6. Render 会自动识别 `render.yaml`
7. 点击部署

## Lovable 里怎么接

你可以直接在 Lovable 里请求：

```text
https://你的-render地址.onrender.com/dashboard
```

返回示例：

```json
{
  "next_hour": 5719.8,
  "tomorrow_total": 83482.8,
  "next_7_days": 580138.3,
  "peak_time": "20:00",
  "peak_value": 5739.6,
  "hourly": [5719.8, 5601.2, 5488.3],
  "generated_at": "2026-04-05T17:30:00Z"
}
```

### 页面字段绑定建议

- 下一小时预计用电 → `next_hour`
- 明日预计总用电 → `tomorrow_total`
- 未来 7 天预计总用电 → `next_7_days`
- 预计高峰时段 → `peak_time`
- 24 小时曲线 → `hourly`

## 可改参数

默认位置：
- `lat=53.34`
- `lon=-6.26`

例如：

```text
https://你的-render地址.onrender.com/dashboard?lat=53.34&lon=-6.26
```

## 重要提醒

因为你的模型是用 `scikit-learn 1.6.1` 保存的，部署时我已经把依赖固定到了这个版本，避免反序列化报错。
