import re
from datetime import date, datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class UserInput(BaseModel):
    name: str = Field(..., description="User's name")
    email: EmailStr = Field(..., description="User's email address")
    query: str = Field(..., description="User's query")
    order_id: Optional[str] = Field(
        None,
        description="Order ID if available (format: ABC-12345)",
    )
    purchase_date: Optional[date] = None

    @field_validator("order_id")
    def validate_order_id(cls, order_id):
        if order_id is None:
            return order_id
        if not re.match(r"^[A-Z]{3}-\d{5}$", order_id):
            raise ValueError(
                "order_id must be in format ABC-12345 "
                "(3 uppercase letters, dash, 5 digits)"
            )
        return order_id


class CustomerQuery(UserInput):
    priority: str = Field(..., description="Priority level: low, medium, high")
    category: Literal[
        "refund_request", "information_request", "other"
    ] = Field(..., description="Query category")
    is_complaint: bool = Field(
        ..., description="Whether this is a complaint"
    )
    tags: List[str] = Field(..., description="Relevant keyword tags")


class FAQLookupArgs(BaseModel):
    query: str = Field(..., description="User's query")
    tags: List[str] = Field(
        ..., description="Relevant keyword tags from the customer query"
    )


class CheckOrderStatusArgs(BaseModel):
    order_id: str = Field(
        ..., description="Customer's order ID (format: ABC-12345)"
    )
    email: EmailStr = Field(..., description="Customer's email address")

    @field_validator("order_id")
    def validate_order_id(cls, order_id):
        if not re.match(r"^[A-Z]{3}-\d{5}$", order_id):
            raise ValueError(
                "order_id must be in format ABC-12345 "
                "(3 uppercase letters, dash, 5 digits)"
            )
        return order_id


class OrderDetails(BaseModel):
    status: str
    estimated_delivery: str
    note: str


class SupportTicket(CustomerQuery):
    recommended_next_action: Literal[
        "escalate_to_agent",
        "send_faq_response",
        "send_order_status",
        "no_action_needed",
    ] = Field(..., description="LLM's recommended next action for support")
    order_details: Optional[OrderDetails] = Field(
        None, description="Order details if action is send_order_status"
    )
    faq_response: Optional[str] = Field(
        None, description="FAQ response if action is send_faq_response"
    )
    creation_date: datetime = Field(
        ..., description="Date and time the ticket was created"
    )
