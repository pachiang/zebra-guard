"""檢舉書草稿 .txt 生成。

格式走純文字,使用者可自行複製到警政署檢舉網頁,或另存為 pdf。留白欄位預設為
`【請填入】`,強制提醒使用者不要直接送出空白內容。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

TEMPLATE = """\
交通違規檢舉書草稿(ZebraGuard 自動產生)

================================================================
一、違規事實
  違規類型:汽(機)車行經行人穿越道未停讓行人通過
  法條依據:道路交通管理處罰條例 第 44 條 第 2 項(請以現行法為準)

二、影片資訊
  影片檔名:{video_name}
  違規事件編號:#{display_index:03d}(本影片共 {total_events} 筆)

三、違規時間
  事件起訖:{start_str} ~ {end_str}
  持續秒數:{duration:.2f} 秒

四、違規地點(【請填入】)
  路口 / 街道名稱:【請填入】
  鄰近地標或門牌:【請填入】
  行政區 / 縣市:【請填入】

五、違規車輛
  車牌號碼:{plate_line}
  車種 / 顏色:【請填入】(本程式自動判斷可能有誤,請以影像為準)

六、事件說明(可保留或修改)
  本車輛於上述時間行經斑馬線時,斑馬線上仍有行人通行,車輛未停讓即通過。
  違規片段檔名為 clip.mp4;關鍵幀檔名為 keyframe_raw.jpg / keyframe_annotated.jpg,
  三者位於本資料夾內。

七、檢舉人資訊(【請填入】)
  姓名:【請填入】
  身分證字號:【請填入】
  聯絡電話 / Email:【請填入】
  通訊地址:【請填入】

八、切結
  本人切結以上資訊屬實,並同意受理機關將本檢舉相關事項通知本人。
  提報日期:【請填入】

================================================================
提醒:
  · 本檔案為 AI 輔助產生的「草稿」,不代表違規事實於法律上成立。
  · 請於提報前親自檢視 clip.mp4 與關鍵幀,確認違規事實與車牌辨識均清楚。
  · 「三枕木紋」距離於本程式以斑馬線 mask 像素接近度近似判定,非精確 3 公尺。
  · 報告產生時間:{generated_at}(ZebraGuard v{app_version})
"""


def _fmt_timestamp(sec: float) -> str:
    m = int(sec) // 60
    s = sec - 60 * m
    h = m // 60
    m = m % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def write_complaint_draft(
    out_path: Path,
    *,
    event: dict[str, Any],
    video_name: str,
    total_events: int,
    app_version: str,
) -> None:
    plate = (event.get("license_plate") or "").strip()
    plate_line = plate if plate else "【請填入】"
    text = TEMPLATE.format(
        video_name=video_name,
        display_index=event.get("_display_index", event.get("id", 0)),
        total_events=total_events,
        start_str=_fmt_timestamp(event["start_sec"]),
        end_str=_fmt_timestamp(event["end_sec"]),
        duration=event["end_sec"] - event["start_sec"],
        plate_line=plate_line,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        app_version=app_version,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
