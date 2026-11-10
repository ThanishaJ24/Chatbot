from fastapi import FastAPI

from app.route import router

app = FastAPI(title="Portfolio Bot API")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Welcome to Portfolio Bot"}


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
