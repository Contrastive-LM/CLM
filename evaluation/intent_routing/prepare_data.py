#!/usr/bin/env python3
"""Build the intent-routing benchmark (Banking77, Bitext insurance) as typed-decisions parquet.

Each customer message becomes one ``choice`` question ("Which support queue should
handle this customer message?") whose options are the dataset's intent labels, each
with a one-sentence hand-written description. Rows use the ``LocalLLaMA/typed-decisions``
schema (``id``, ``workflow``, ``state``, ``questions``, ``gold``), so they go straight into
``train/finetune.py --task choice`` via ``train/adapters.py:typed_decision_examples``.

Splits (per dataset, all randomness seeded with 0)
  * text is stripped; exact duplicate texts are dropped, and a text found under two
    labels is dropped entirely
  * round(0.2 * n_labels) labels are held out as *unseen*; none of their messages are
    in train
  * every label is split 50/50 into train / test; each part is then subsampled,
    stratified by label, to at most 5000 train / 1500 test-seen / 1000 test-unseen rows

Output (``--out``, default data/intent_routing)
  <name>/seen/train-00000-of-00001.parquet     train (seen labels)
  <name>/seen/test-00000-of-00001.parquet      test, seen labels
  <name>/unseen/train-00000-of-00001.parquet   same train rows
  <name>/unseen/test-00000-of-00001.parquet    test, unseen labels
  <name>/labels.json                           question, label descriptions, unseen labels, source

Train questions list the seen labels only, so unseen label descriptions never reach
training, not even as negatives. Test questions list every label, seen and unseen.

Sources are pinned; the Bitext data (CDLA-Sharing-1.0) is downloaded, not redistributed.

  python evaluation/intent_routing/prepare_data.py --out data/intent_routing
  python train/finetune.py --task choice --data data/intent_routing/banking77 --workflow seen \\
      --init-ckpt "$(clm-download)" --out-dir runs/banking77_seen
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import urllib.request

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SEED = 0
UNSEEN_FRAC = 0.2
CAP_TRAIN, CAP_TEST_SEEN, CAP_TEST_UNSEEN = 5000, 1500, 1000
QID = "queue"
QUESTION = "Which support queue should handle this customer message?"

# Pinned sources. PolyAI/banking77 on the Hub is a loading script that reads these CSVs.
B77_REV = "57ec275d8078af65b7731c2a98be812d844a6d6b"
B77_URLS = {s: f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{B77_REV}/banking_data/{s}.csv"
            for s in ("train", "test")}
BITEXT_REPO, BITEXT_REV = "bitext/Bitext-insurance-llm-chatbot-training-dataset", "3ce42aaa134857c47c7ee6ad821be1d1fcf5a75b"
BITEXT_URL = (f"https://huggingface.co/datasets/{BITEXT_REPO}/resolve/{BITEXT_REV}/"
              "bitext-insurance-llm-chatbot-training-dataset.csv")

# One sentence per label, written by hand from the label name and a few examples.
B77_DESC = {
    "Refund_not_showing_up": "Customer asks why a refund they expected has not appeared on their account.",
    "activate_my_card": "Customer wants to activate their card or has trouble activating it.",
    "age_limit": "Customer asks about the minimum age needed to open or use an account.",
    "apple_pay_or_google_pay": "Customer asks about using Apple Pay or Google Pay, including top-ups through them.",
    "atm_support": "Customer asks where they can find an ATM or which ATMs they can use.",
    "automatic_top_up": "Customer asks about setting up or using automatic top-ups.",
    "balance_not_updated_after_bank_transfer": "Customer says their balance has not updated after a bank transfer.",
    "balance_not_updated_after_cheque_or_cash_deposit": "Customer says their balance has not updated after depositing a cheque or cash.",
    "beneficiary_not_allowed": "Customer asks why they cannot add or pay a beneficiary.",
    "cancel_transfer": "Customer wants to cancel or reverse a transfer they made.",
    "card_about_to_expire": "Customer asks what to do because their card is about to expire.",
    "card_acceptance": "Customer asks where their card is accepted.",
    "card_arrival": "Customer says an ordered card has not arrived yet.",
    "card_delivery_estimate": "Customer asks how long card delivery will take.",
    "card_linking": "Customer wants to link a card to their account or app.",
    "card_not_working": "Customer says their physical card is not working.",
    "card_payment_fee_charged": "Customer asks about a fee charged on a card payment.",
    "card_payment_not_recognised": "Customer sees a card payment they do not recognise.",
    "card_payment_wrong_exchange_rate": "Customer thinks the wrong exchange rate was applied to a card payment.",
    "card_swallowed": "Customer says an ATM kept their card.",
    "cash_withdrawal_charge": "Customer asks about a fee charged on a cash withdrawal.",
    "cash_withdrawal_not_recognised": "Customer sees a cash withdrawal they did not make.",
    "change_pin": "Customer wants to change their card PIN.",
    "compromised_card": "Customer thinks their card details have been stolen or misused.",
    "contactless_not_working": "Customer says contactless payments are not working.",
    "country_support": "Customer asks which countries the service is available in.",
    "declined_card_payment": "Customer says a card payment was declined.",
    "declined_cash_withdrawal": "Customer says a cash withdrawal was declined.",
    "declined_transfer": "Customer says a transfer was declined.",
    "direct_debit_payment_not_recognised": "Customer sees a direct debit they do not recognise.",
    "disposable_card_limits": "Customer asks about the limits on disposable cards.",
    "edit_personal_details": "Customer wants to change their personal details.",
    "exchange_charge": "Customer asks about fees for exchanging currency.",
    "exchange_rate": "Customer asks what exchange rate is used and how it is set.",
    "exchange_via_app": "Customer wants to exchange currency in the app.",
    "extra_charge_on_statement": "Customer asks about an unexplained extra charge on their statement.",
    "failed_transfer": "Customer says a transfer failed.",
    "fiat_currency_support": "Customer asks which fiat currencies can be held or exchanged.",
    "get_disposable_virtual_card": "Customer wants to get a disposable virtual card.",
    "get_physical_card": "Customer asks how to get their physical card or where to find its PIN.",
    "getting_spare_card": "Customer wants an extra card for their account.",
    "getting_virtual_card": "Customer wants to get a virtual card.",
    "lost_or_stolen_card": "Customer says their card is lost or stolen.",
    "lost_or_stolen_phone": "Customer says their phone is lost or stolen.",
    "order_physical_card": "Customer wants to order a physical card or asks what it costs.",
    "passcode_forgotten": "Customer has forgotten their passcode and wants to reset it.",
    "pending_card_payment": "Customer asks why a card payment is still pending.",
    "pending_cash_withdrawal": "Customer asks why a cash withdrawal is still pending.",
    "pending_top_up": "Customer asks why a top-up is still pending.",
    "pending_transfer": "Customer asks why a transfer is still pending.",
    "pin_blocked": "Customer says their PIN is blocked and wants it unblocked.",
    "receiving_money": "Customer asks how to receive money into their account.",
    "request_refund": "Customer wants a refund for a purchase.",
    "reverted_card_payment?": "Customer asks why a card payment was reverted or returned.",
    "supported_cards_and_currencies": "Customer asks which cards and currencies are supported.",
    "terminate_account": "Customer wants to close their account.",
    "top_up_by_bank_transfer_charge": "Customer asks about fees for topping up by bank transfer.",
    "top_up_by_card_charge": "Customer asks about fees for topping up by card.",
    "top_up_by_cash_or_cheque": "Customer wants to top up with cash or a cheque.",
    "top_up_failed": "Customer says a top-up failed.",
    "top_up_limits": "Customer asks about limits on top-ups.",
    "top_up_reverted": "Customer asks why a top-up was reverted.",
    "topping_up_by_card": "Customer wants to top up their account with a card.",
    "transaction_charged_twice": "Customer says they were charged twice for one transaction.",
    "transfer_fee_charged": "Customer asks about a fee charged on a transfer.",
    "transfer_into_account": "Customer wants to transfer money into their account.",
    "transfer_not_received_by_recipient": "Customer says the recipient has not received a transfer.",
    "transfer_timing": "Customer asks how long a transfer takes.",
    "unable_to_verify_identity": "Customer says they cannot complete identity verification.",
    "verify_my_identity": "Customer asks how to verify their identity and what documents are needed.",
    "verify_source_of_funds": "Customer asks about verifying the source of their funds.",
    "verify_top_up": "Customer asks about verifying a top-up or its verification code.",
    "virtual_card_not_working": "Customer says their virtual card is not working.",
    "visa_or_mastercard": "Customer asks whether they can get a Visa or Mastercard card.",
    "why_verify_identity": "Customer asks why identity verification is required.",
    "wrong_amount_of_cash_received": "Customer says an ATM gave them the wrong amount of cash.",
    "wrong_exchange_rate_for_cash_withdrawal": "Customer thinks the wrong exchange rate was applied to a cash withdrawal.",
}

BITEXT_DESC = {
    "accept_settlement": "Customer wants to accept a settlement offer.",
    "agent": "Customer wants to speak with the insurance company.",
    "appeal_denied_insurance_claim": "Customer wants to appeal a denied insurance claim.",
    "buy_insurance_policy": "Customer wants to buy an insurance policy.",
    "calculate_insurance_quote": "Customer wants a quote for an insurance policy.",
    "cancel_insurance_policy": "Customer wants to cancel their insurance policy.",
    "cancellation_fees": "Customer asks about cancellation or early termination fees.",
    "change_coverage": "Customer wants to change the coverage on their policy.",
    "change_personal_details": "Customer wants to change the personal details on their policy.",
    "check_coverage": "Customer wants to check what their policy covers.",
    "check_payments": "Customer wants to see the payments they have made.",
    "check_rates": "Customer asks about insurance rates.",
    "compare_insurance_policies": "Customer wants to compare different insurance policies.",
    "customer_service": "Customer wants to contact customer service.",
    "dispute_invoice": "Customer wants to dispute an invoice or a charge on their bill.",
    "downgrade_coverage": "Customer wants to reduce the coverage on their policy.",
    "file_claim": "Customer wants to file an insurance claim.",
    "file_complaint": "Customer wants to make a complaint about the service.",
    "general_information": "Customer asks for general information about the insurer or its policies.",
    "human_agent": "Customer wants to talk to a human agent.",
    "information_auto_insurance": "Customer asks about their auto insurance.",
    "information_health_insurance": "Customer asks about their health insurance.",
    "information_home_insurance": "Customer asks about their home insurance.",
    "information_life_insurance": "Customer asks about their life insurance.",
    "information_pet_insurance": "Customer asks about their pet insurance.",
    "information_travel_insurance": "Customer asks about their travel insurance.",
    "insurance_representative": "Customer wants to speak with their insurance representative.",
    "negotiate_settlement": "Customer wants to negotiate a settlement offer.",
    "pay": "Customer wants to make a payment.",
    "payment_methods": "Customer asks which payment methods they can use.",
    "receive_payment": "Customer wants to receive a compensation payment.",
    "reject_settlement": "Customer wants to reject a settlement offer.",
    "renew_insurance_policy": "Customer wants to renew their insurance policy.",
    "report_incident": "Customer wants to report an incident.",
    "report_payment_issue": "Customer wants to report a problem making a payment.",
    "schedule_appointment": "Customer wants to book an appointment with a professional.",
    "schedule_payments": "Customer wants to schedule their payments.",
    "track_claim": "Customer wants to check the status of their claim.",
    "upgrade_coverage": "Customer wants to increase the coverage on their policy.",
}

def fetch(url: str, raw_dir: str, name: str) -> str:
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, name)
    if not os.path.exists(path):
        print(f"[data] download {url}", flush=True)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def stratified_cap(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Proportional per-label subsample to at most ``cap`` rows (largest remainder), keeping row order."""
    if len(df) <= cap:
        return df
    counts = df["label"].value_counts().sort_index()
    quota = counts * cap / len(df)
    k = quota.apply(math.floor)
    rem = (quota - k).sort_values(ascending=False, kind="stable")
    for lab in rem.index[: cap - int(k.sum())]:
        k[lab] += 1
    keep = [g.index[: k[lab]] for lab, g in df.groupby("label", sort=True)]  # df is already shuffled per label
    return df.loc[sorted(i for idx in keep for i in idx)]


def split_intents(name: str, df: pd.DataFrame, labels: list[str]):
    """df: columns text, label (source order) -> (deduplicated df, unseen, train, test_seen, test_unseen)."""
    df = df.assign(text=df["text"].astype(str).str.strip())
    # drop exact duplicate texts (all copies if labels conflict) so no text is in both train and test
    n_lab = df.groupby("text")["label"].nunique()
    df = df[df["text"].map(n_lab) == 1].drop_duplicates("text").reset_index(drop=True)
    df["id"] = [f"{name}-{i:05d}" for i in range(len(df))]
    found = sorted(df["label"].unique())
    assert found == labels, set(found) ^ set(labels)
    unseen = sorted(random.Random(SEED).sample(labels, round(UNSEEN_FRAC * len(labels))))

    rng = random.Random(SEED)
    tr, te = [], []
    for lab in labels:
        idx = list(df.index[df["label"] == lab])
        rng.shuffle(idx)
        h = len(idx) // 2
        tr += idx[:h]
        te += idx[h:]
    # keep shuffled-within-label order so stratified_cap takes a random subset per label
    train_all, test_all = df.loc[tr], df.loc[te]
    train = stratified_cap(train_all[~train_all["label"].isin(unseen)], CAP_TRAIN)
    test_seen = stratified_cap(test_all[~test_all["label"].isin(unseen)], CAP_TEST_SEEN)
    test_unseen = stratified_cap(test_all[test_all["label"].isin(unseen)], CAP_TEST_UNSEEN)
    assert not set(train["label"]) & set(unseen)
    assert not set(train["text"]) & (set(test_seen["text"]) | set(test_unseen["text"]))
    return df, unseen, train, test_seen, test_unseen


def write_rows(path: str, name: str, df: pd.DataFrame, criteria: dict[str, str]) -> None:
    """typed-decisions rows: state is the message; one choice question over ``criteria``."""
    questions = json.dumps({QID: {"type": "choice", "instructions": QUESTION, "criteria": criteria}},
                           ensure_ascii=False)
    table = pa.table({
        "id": list(df["id"]),
        "workflow": [name] * len(df),
        "state": list(df["text"]),
        "questions": [questions] * len(df),
        "gold": [json.dumps({QID: {"label": lab}}, ensure_ascii=False) for lab in df["label"]],
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path)


def build(name: str, df: pd.DataFrame, desc: dict[str, str], source: dict, out: str) -> dict:
    labels = sorted(desc)
    df, unseen, train, test_seen, test_unseen = split_intents(name, df, labels)
    all_opts = {lab: desc[lab] for lab in labels}
    seen_opts = {lab: d for lab, d in all_opts.items() if lab not in unseen}
    d = os.path.join(out, name)
    for wf, test in (("seen", test_seen), ("unseen", test_unseen)):
        write_rows(os.path.join(d, wf, "train-00000-of-00001.parquet"), name, train, seen_opts)
        write_rows(os.path.join(d, wf, "test-00000-of-00001.parquet"), name, test, all_opts)
    with open(os.path.join(d, "labels.json"), "w") as f:
        json.dump({"question": QUESTION, "labels": all_opts, "unseen": unseen, "source": source},
                  f, indent=1, ensure_ascii=False)
    return {"labels": len(labels), "unseen": len(unseen), "rows_after_dedup": len(df),
            "train": len(train), "test_seen": len(test_seen), "test_unseen": len(test_unseen)}


def banking77(out: str, raw_dir: str) -> dict:
    df = pd.concat([pd.read_csv(fetch(u, raw_dir, f"banking77_{s}.csv")) for s, u in B77_URLS.items()],
                   ignore_index=True)
    df = df.rename(columns={"category": "label"})[["text", "label"]]
    return build("banking77", df, B77_DESC,
                 {"dataset": "PolyAI/banking77", "licence": "CC-BY-4.0", "files": list(B77_URLS.values()),
                  "note": "upstream train+test pooled, then re-split"}, out)


def bitext(out: str, raw_dir: str) -> dict:
    df = pd.read_csv(fetch(BITEXT_URL, raw_dir, "bitext_insurance.csv"))
    df = df.rename(columns={"instruction": "text", "intent": "label"})[["text", "label"]]
    return build("bitext", df, BITEXT_DESC,
                 {"dataset": BITEXT_REPO, "revision": BITEXT_REV, "licence": "CDLA-Sharing-1.0",
                  "text_field": "instruction", "label_field": "intent"}, out)


DATASETS = {"banking77": banking77, "bitext": bitext}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join("data", "intent_routing"), help="output root")
    ap.add_argument("--raw-cache", default=None, help="where source CSVs are downloaded (default OUT/raw)")
    ap.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    args = ap.parse_args()
    raw_dir = args.raw_cache or os.path.join(args.out, "raw")
    stats = {name: DATASETS[name](args.out, raw_dir) for name in args.datasets}
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
