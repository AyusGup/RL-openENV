"""A FastAPI application for item management."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str = "item"

db = []

@app.post("/api/items")
def create_item(item: Item):
    """Create an item and return a JSONResponse.
    """
    db.append(item.dict())
    return JSONResponse(
        content={"id": len(db), "item": item.dict()},
        status_code=200
    )

@app.get("/api/items/{item_id}")
def get_item(item_id: int):
    if 1 <= item_id <= len(db):
        return db[item_id - 1]
    raise HTTPException(status_code=404, detail="Item not found")
