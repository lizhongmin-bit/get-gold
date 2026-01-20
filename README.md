# tail30_selector

“尾盘30分钟选股法”量化复刻工具，提供可配置的筛选链路、解释性输出与基础回测。

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
python -m tail30_selector --date 2026-01-20 --mode realtime --universe all --datasource akshare
```

- `--mode realtime|backtest`
- `--universe all|hs300|custom`
- `--datasource akshare|tushare`

### 输出

- 控制台打印 Top 20 结果与触发原因。
- CSV 输出至 `output/selected_YYYYMMDD.csv`。

## 策略链路（Step1~Step7）

- Step1: 14:30后涨幅 3%-5%。
- Step2: 量比 >= 1。
- Step3: 换手率 5%-10%。
- Step4: 流通市值 50亿-100亿。
- Step5: 成交量台阶式放量（趋势上升、分段均值递增，排除心电图式）。
- Step6: 均线多头排列 + 均线上方 + 上升发散。
- Step7: 分时强势 + 跑赢大盘 + 14:30后创新高 + 回踩不破。

## 数据源

- AkShare：优先使用，支持 A 股现货、分钟线、指数分钟线。
- Tushare：备选，需要 `TS_TOKEN`（或 `--token`），分钟线可能受限。

## 风险提示

策略不是100%胜率，只是提高概率；有时候筛完一只都没有 => 空仓也是操作；
要设置止盈止损，行情不对及时撤退；仅供学习交流，不构成投资建议。
