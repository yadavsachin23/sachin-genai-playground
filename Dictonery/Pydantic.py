# Pydantic is a powerful data validation and settings management library for Python. It allows you to define data models with type annotations, and it will automatically validate and parse the data according to those annotations. This can be particularly useful when working with APIs, databases, or any situation where you need to ensure that the data you're working with conforms to a specific structure.

from pydantic import BaseModel, EmailStr
from typing import Optional


class User(BaseModel):
    id: int
    name: str
    email: EmailStr
    age: Optional[int] = None


new_user = {"id": 1, "name": "Alice", "email": "alice@gmail.com", "age": 30}

student = User(**new_user)
print(student)
