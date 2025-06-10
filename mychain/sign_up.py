from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from utils.get_chroma import get_or_create_user_chromadb

router = APIRouter()


class SignupRequest(BaseModel):
    user_id: str


@router.post("/signup")
def signup(req: SignupRequest) -> JSONResponse:
    try:
        get_or_create_user_chromadb(req.user_id)
        return JSONResponse(content={"success": True})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)})
