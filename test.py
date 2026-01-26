# ======================
# Tushare Token: 复制下面第4到7行代码，粘贴到你的项目中
# ======================
import tushare as ts
import tushare.pro.client as client
client.DataApi._DataApi__http_url = "http://tushare.xyz:5000"
pro = ts.pro_api('1d75857c8ae77369e6314c076d8cc0cdfefde3f3fd0431e179ed916e')
# -----------------------------

# ======================
# 10000积分账号测试 - CCASS港股持股明细
# ======================
# 获取港股中央结算系统(CCASS)持股明细数据
# 此接口需要8000积分以上才能使用

df = pro.ccass_hold_detail(**{
    "ts_code": "",
    "trade_date": "",
    "start_date": "",
    "end_date": "",
    "hk_code": "",
    "offset": "",
    "limit": 20
}, fields=[
    "trade_date",
    "ts_code",
    "name",
    "col_participant_id",
    "col_participant_name",
    "col_shareholding",
    "col_shareholding_percent"
])
print("📊 CCASS港股持股明细数据：")
print(df)