import os
import time
import uuid

import cv2
import gradio as gr

import config
from classifier import Classifier
from products import PRODUCTS
from scan_fsm import ScanFSM
from smoother import ProbSmoother

clf = Classifier()  # 서버 시작 때 한 번만 로드 (읽기 전용이라 접속자끼리 공유해도 안전)

GRAY, YELLOW, GREEN = (160, 160, 160), (255, 200, 0), (60, 200, 100)  # RGB 기준으로 그린다


def card_md(label):
    if label is None:
        return "### 아직 인식된 상품이 없어요\n상품을 카메라에 비춰 주세요."
    p = PRODUCTS[label]
    return f"### {p['name']}\n**{p['price']:,}원**\n\n{p['desc']}"


def cart_md(cart):
    if not cart:
        return "🛒 장바구니가 비어 있어요."
    lines, total = [], 0
    for k, q in cart.items():
        sub = PRODUCTS[k]["price"] * q
        total += sub
        lines.append(f"- {PRODUCTS[k]['name']} × {q} = {sub:,}원")
    return "🛒 **장바구니**\n" + "\n".join(lines) + f"\n\n**합계 {total:,}원**"


def candidate(probs):
    """상품 후보 이름 또는 None. 'none' 클래스·낮은 확률·미등록 상품은 모두 None."""
    label, conf = clf.top(probs)
    if label == config.NONE_LABEL or conf < config.CONF_MIN or label not in PRODUCTS:
        return None, label, conf
    return label, label, conf


def scan(rgb, st):
    """스트리밍 프레임 1장 처리. 타이머·평균은 접속자별 gr.State(st)에만 둔다."""
    if st is None:
        st = {"fsm": ScanFSM(), "sm": ProbSmoother(config.SMOOTH_N), "current": None}
    if rgb is None:  # 카메라 권한 거부 / 프레임 없음
        return None, "📷 카메라 권한을 허용해 주세요.", card_md(st["current"]), st

    cand, _, conf = candidate(st["sm"].update(clf.predict(rgb)))
    done, ratio = st["fsm"].update(cand, time.time())
    if done:
        st["current"] = done

    if done:
        msg, color = f"✅ {PRODUCTS[done]['name']} 인식 완료!", GREEN
    elif cand:
        msg, color = f"🔍 {PRODUCTS[cand]['name']}? 잠깐만 그대로... ({conf:.0%})", YELLOW
    else:
        msg, color = "상품을 화면 가운데에 비춰 주세요.", GRAY

    out = rgb.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, 6)                       # 상태 테두리
    cv2.rectangle(out, (0, h - 14), (int(w * ratio), h), GREEN, -1)            # 진행률 게이지
    return out, msg, card_md(st["current"]), st


def scan_photo(rgb, st):
    """실시간이 잘 안 되는 폰을 위한 대안: 사진 한 장으로 인식 (폰에서는 카메라 앱이 열린다)."""
    if st is None:
        st = {"fsm": ScanFSM(), "sm": ProbSmoother(config.SMOOTH_N), "current": None}
    if rgb is None:
        return "", card_md(st["current"]), st
    cand, label, conf = candidate(clf.predict(rgb))
    if cand:
        st["current"] = cand
        return f"✅ {PRODUCTS[cand]['name']} ({conf:.0%})", card_md(cand), st
    return "❓ 상품을 찾지 못했어요. 더 가까이, 밝은 곳에서 다시 찍어 주세요.", card_md(st["current"]), st


def add_item(st, cart):  # st 는 읽기만 한다 (스트림과 상태 덮어쓰기 충돌 방지)
    cart = dict(cart or {})
    if st and st.get("current"):
        cur = st["current"]
        cart[cur] = cart.get(cur, 0) + 1
    return cart, cart_md(cart)


def pay(cart):
    if not cart:
        return cart, cart_md(cart), "담긴 상품이 없어요."
    total = sum(PRODUCTS[k]["price"] * q for k, q in cart.items())
    receipt = (f"✅ **모의 결제 완료** (주문번호 {uuid.uuid4().hex[:8]}) — {total:,}원\n\n"
               "※ 연습용입니다. 실제 결제는 이뤄지지 않았어요.")
    return {}, cart_md({}), receipt


with gr.Blocks(title=config.STORE_NAME) as demo:
    gr.Markdown(f"# 🏪 {config.STORE_NAME}\n상품을 카메라에 비추면 정보가 나와요. "
                "영상은 저장되지 않고 화면 안에서만 판정합니다.")
    scan_state = gr.State(None)   # 접속자별 스캔 상태
    cart_state = gr.State({})     # 접속자별 장바구니 (스트림이 건드리지 않는다)

    with gr.Tab("실시간 스캔"):
        cam = gr.Image(sources=["webcam"], streaming=True, type="numpy",
                       mirror_webcam=False, label="카메라 (후면 카메라를 선택하세요)")
        view = gr.Image(label="인식 화면")
        status = gr.Markdown()
    with gr.Tab("사진으로 인식"):
        photo = gr.Image(sources=["upload"], type="numpy", label="사진 찍기 / 선택")
        photo_msg = gr.Markdown()

    card = gr.Markdown(card_md(None))
    with gr.Row():
        add_btn = gr.Button("장바구니 담기", variant="primary")
        pay_btn = gr.Button("결제하기 (모의)")
        clear_btn = gr.Button("비우기")
    cart_view = gr.Markdown(cart_md({}))
    receipt = gr.Markdown()

    cam.stream(scan, [cam, scan_state], [view, status, card, scan_state],
               stream_every=config.STREAM_EVERY)
    photo.change(scan_photo, [photo, scan_state], [photo_msg, card, scan_state])
    add_btn.click(add_item, [scan_state, cart_state], [cart_state, cart_view])
    pay_btn.click(pay, cart_state, [cart_state, cart_view, receipt])
    clear_btn.click(lambda: ({}, cart_md({})), None, [cart_state, cart_view])

if __name__ == "__main__":
    # 배포(Spaces)에서는 인자 없이 launch(). 폰 테스트는 SHARE=1 로 https 임시 링크를 받는다.
    demo.launch(share=os.getenv("SHARE") == "1")
