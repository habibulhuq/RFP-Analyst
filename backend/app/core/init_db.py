from .db import Base, engine
from ..models import dataset, job, chat_session

def init_db():
    # Import all models so SQLAlchemy knows them
    Base.metadata.create_all(bind=engine)
