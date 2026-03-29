"""
Router: /coins
Shared gamification state for the hardcoded user.
"""

from fastapi import APIRouter, HTTPException, Query

from app.models import (
    CoinStateResponse,
    CoinsBootstrapRequest,
    DailyLoginRequest,
    DailyLoginResponse,
    MissionCompleteRequest,
    MissionCompleteResponse,
    ReferralApplyRequest,
    ReferralApplyResponse,
)
from app.services.coins_service import (
    apply_referral_code,
    bootstrap_coin_state,
    claim_daily_login,
    complete_mission,
    get_coin_state,
)

router = APIRouter(prefix="/coins", tags=["Coins"])


@router.get("", response_model=CoinStateResponse)
async def get_coins(user_id: str = Query(...)):
    try:
        return await get_coin_state(user_id)
    except Exception as exc:
        print(f"[coins/get] Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/bootstrap", response_model=CoinStateResponse)
async def bootstrap_coins(request: CoinsBootstrapRequest):
    try:
        legacy_state = request.legacy_state.model_dump() if request.legacy_state else None
        return await bootstrap_coin_state(request.user_id, legacy_state=legacy_state)
    except Exception as exc:
        print(f"[coins/bootstrap] Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/daily-login", response_model=DailyLoginResponse)
async def daily_login(request: DailyLoginRequest):
    try:
        coin_state, reward = await claim_daily_login(request.user_id)
        return {"coin_state": coin_state, "reward": reward}
    except Exception as exc:
        print(f"[coins/daily-login] Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/missions/complete", response_model=MissionCompleteResponse)
async def complete_coin_mission(request: MissionCompleteRequest):
    try:
        coin_state, earned_amount = await complete_mission(
            user_id=request.user_id,
            mission_id=request.mission_id,
        )
        return {"coin_state": coin_state, "earned_amount": earned_amount}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        print(f"[coins/mission-complete] Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/referral/apply", response_model=ReferralApplyResponse)
async def apply_referral(request: ReferralApplyRequest):
    try:
        coin_state, result = await apply_referral_code(
            user_id=request.user_id,
            code=request.code,
        )
        return {"coin_state": coin_state, **result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        print(f"[coins/referral-apply] Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
