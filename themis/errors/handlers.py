from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from themis.errors.exceptions import EmptyPetitionError, JudgeParseError


async def empty_petition_handler(request: Request, exc: EmptyPetitionError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def judge_parse_handler(request: Request, exc: JudgeParseError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": "Judge LLM returned an unparseable response."})


def register(app: FastAPI) -> None:
    app.add_exception_handler(EmptyPetitionError, empty_petition_handler)
    app.add_exception_handler(JudgeParseError, judge_parse_handler)
