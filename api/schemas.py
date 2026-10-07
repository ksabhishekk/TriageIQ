"""What goes in and what comes out — Pydantic checks every request before our code runs (bad input -> 422)."""
from datetime import date
from typing import Literal
from pydantic import BaseModel, Field, field_validator

EXAMPLE = {
    "company": "JPMORGAN CHASE & CO.",
    "product": "Checking or savings account",
    "sub_product": "Checking account",
    "issue": "Managing an account",
    "state": "CA",
    "older_american": False,
    "servicemember": False,
    "narrative": ("On XX/XX/XXXX I noticed two overdraft fees of {$35.00} on my checking account. The deposit that "
                  "covered the purchase had posted that same morning, so my balance was never negative. I called "
                  "customer service twice and was told the fees were correct and would not be refunded. I am asking "
                  "the bank to reverse both charges and explain why they were applied."),
}


class ComplaintIn(BaseModel):
    """A new complaint, the way the CFPB form collects it. Names, not ids — see the /options endpoints."""
    model_config = {"json_schema_extra": {"examples": [EXAMPLE]}}

    company: str = Field(min_length=2, max_length=200, description="Company name as in the CFPB data "
                         "(GET /options/companies?search=...). An unknown name is scored as a company with no history.")
    product: str = Field(max_length=200, description="GET /options/products")
    sub_product: str = Field(max_length=200, description="Must belong to the product (GET /options/products)")
    issue: str = Field(max_length=300, description="GET /options/issues")
    state: str | None = Field(default=None, max_length=50, description="2-letter code, or leave empty")
    older_american: bool = Field(default=False, description="The consumer ticked 'Older American' (62+)")
    servicemember: bool = Field(default=False, description="The consumer ticked 'Servicemember'")
    narrative: str = Field(min_length=1, max_length=20_000, description="The complaint in the consumer's words. "
                           "The model learned from CFPB-published text: amounts as {$35.00}, private details as XXXX.")

    @field_validator("narrative")
    @classmethod
    def enough_words(cls, v: str) -> str:
        if sum(w.isalpha() for w in v.split()) < 20:      # training used only complaints with >= 20 real words
            raise ValueError("write at least 20 words: the model was trained on complaints of 20+ real words")
        return v


class Prediction(BaseModel):
    payout_probability: float = Field(description="Chance the company pays (calibrated on the newest data, Jul–Dec 2024)")
    route: Literal["senior analyst", "template response"]
    senior_threshold: float = Field(description="Probability at or above which a complaint goes to a senior analyst "
                                                "(the riskiest 10% of 2024 complaints)")
    model_version: str
    features_as_of: date = Field(description="The day the track-record inputs describe")
    inputs: dict[str, float] = Field(description="The 19 numbers the model read next to the text (from SQL)")
    text_word_pieces: int
    text_cut_at_512: bool = Field(description="True if the text was longer than the model reads (first 510 pieces used)")
    notes: list[str]
    latency_ms: float


class ComplaintOut(Prediction):
    complaint_id: int
    date_received: date
    company: str
    product: str
    sub_product: str
    issue: str
    state: str
    narrative: str = Field(description="Cleaned text, exactly as the model read it ([DATE] / [REDACTED] = CFPB blanks)")
    what_actually_happened: str | None = Field(description="The company's real response")
    actually_paid: bool


class XAIStatus(BaseModel):
    enabled: bool
    reason: str


class FeatureExplanation(BaseModel):
    feature: str
    value: float
    logit_contribution: float


class SentenceExplanation(BaseModel):
    text: str
    logit_contribution: float


class ModalityExplanation(BaseModel):
    text_logit_contribution: float
    track_record_logit_contribution: float
    baseline_logit: float
    current_logit: float
    additivity_error: float


class ExplanationOut(BaseModel):
    payout_probability: float
    route: Literal["senior analyst", "template response"]
    model_version: str
    features_as_of: date
    text_word_pieces: int
    text_cut_at_512: bool
    modality: ModalityExplanation
    features: list[FeatureExplanation]
    feature_baseline_logit: float
    feature_additivity_error: float
    sentences: list[SentenceExplanation]
    sentence_additivity_error: float
    latency_ms: float
    caveat: str = "Attributions describe model behavior, not the cause of a real-world payout."
