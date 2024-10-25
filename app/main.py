from fastapi import FastAPI
from .routes.route import router

app = FastAPI()

# Include the routes
app.include_router(router)
