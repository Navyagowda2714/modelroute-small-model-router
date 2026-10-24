"""Label taxonomy for support tickets: intent -> category, base priority, description."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Priority = Literal["low", "medium", "high", "urgent"]
PRIORITIES: list[str] = ["low", "medium", "high", "urgent"]


class IntentSpec(BaseModel):
    name: str
    category: str
    base_priority: Priority
    description: str


INTENTS: dict[str, IntentSpec] = {
    spec.name: spec
    for spec in [
        # billing
        IntentSpec(name="billing_double_charge", category="billing", base_priority="high",
                   description="Customer was charged more than once for the same thing."),
        IntentSpec(name="refund_request", category="billing", base_priority="medium",
                   description="Customer wants money back for a purchase or renewal."),
        IntentSpec(name="invoice_request", category="billing", base_priority="low",
                   description="Customer needs an invoice, receipt or VAT document."),
        IntentSpec(name="payment_failed", category="billing", base_priority="high",
                   description="A payment or card was declined / could not be processed."),
        IntentSpec(name="update_payment_method", category="billing", base_priority="low",
                   description="Customer wants to change the card or payment method on file."),
        # account
        IntentSpec(name="login_issue", category="account", base_priority="medium",
                   description="Customer cannot sign in (errors, 2FA, SSO), password is known."),
        IntentSpec(name="password_reset", category="account", base_priority="medium",
                   description="Customer forgot the password or the reset email/link fails."),
        IntentSpec(name="account_locked", category="account", base_priority="high",
                   description="Account was locked, suspended or disabled."),
        IntentSpec(name="update_account_details", category="account", base_priority="low",
                   description="Change name, email, phone or other profile details."),
        IntentSpec(name="delete_account", category="account", base_priority="medium",
                   description="Customer wants the account and data permanently deleted."),
        # subscription
        IntentSpec(name="cancel_subscription", category="subscription", base_priority="medium",
                   description="Customer wants to stop the subscription / auto-renewal."),
        IntentSpec(name="upgrade_plan", category="subscription", base_priority="low",
                   description="Customer wants a higher tier or more seats."),
        IntentSpec(name="downgrade_plan", category="subscription", base_priority="low",
                   description="Customer wants a cheaper tier or fewer seats, staying subscribed."),
        # shipping
        IntentSpec(name="shipping_delay", category="shipping", base_priority="medium",
                   description="An order is late or has not arrived by the promised date."),
        IntentSpec(name="track_order", category="shipping", base_priority="low",
                   description="Customer asks where an order is / wants tracking information."),
        IntentSpec(name="wrong_item", category="shipping", base_priority="medium",
                   description="Customer received a different item than ordered."),
        IntentSpec(name="damaged_item", category="shipping", base_priority="medium",
                   description="The item arrived broken or damaged."),
        IntentSpec(name="change_shipping_address", category="shipping", base_priority="medium",
                   description="Customer wants to change the delivery address of an order."),
        # technical
        IntentSpec(name="app_crash", category="technical", base_priority="high",
                   description="The app/site crashes, freezes or shows errors."),
        IntentSpec(name="feature_request", category="technical", base_priority="low",
                   description="Customer suggests a new feature or improvement."),
    ]
}

INTENT_NAMES: list[str] = sorted(INTENTS)
CATEGORIES: list[str] = sorted({s.category for s in INTENTS.values()})


def category_of(intent: str) -> str:
    return INTENTS[intent].category


def shift_priority(priority: str, delta: int) -> str:
    idx = PRIORITIES.index(priority) + delta
    return PRIORITIES[max(0, min(len(PRIORITIES) - 1, idx))]
