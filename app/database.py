from sqlalchemy import Column, Integer, Boolean, ForeignKey, Date,String
from sqlalchemy.orm import declarative_base

    
Base = declarative_base()
class MedicineEffectiveness(Base):
    __tablename__ = "medicine_effectiveness"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    medicine_id = Column(Integer, nullable=False)
    side_effect_occurred = Column(Boolean, nullable=False)
    side_effects = Column(String(255), nullable=True)