"""python -m src.make_sample_pdf -> data/sample_policy.pdf (3 pages; no remote-work policy on purpose)."""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

PAGES = [
 ("Company Leave Policy", [
  "<b>1. Annual leave.</b> Employees are eligible for 12 paid leaves per year. Leave accrues at one day per month and is credited on the first working day of each month. New joiners receive a pro-rated balance based on their joining date.",
  "<b>2. Sick leave.</b> Employees receive 6 paid sick days per year. All sick leave requires manager approval, and absences longer than two consecutive days must be supported by a medical certificate submitted to HR within 3 days of returning to work.",
  "<b>3. Carry-over.</b> Up to 5 unused annual leave days may be carried over to the next calendar year. Any additional unused days expire on 31 March and are not paid out.",
  "<b>4. Requesting leave.</b> Planned leave must be requested in the HR portal at least 7 days in advance. Managers respond within 3 working days. Leave during the year-end closing period requires director approval.",
  "<b>5. Parental leave.</b> Employees receive 26 weeks of paid maternity leave and 2 weeks of paid paternity leave. Requests must be submitted at least 30 days before the expected start date, except in emergencies."]),
 ("Refund and Return Policy", [
  "<b>1. Return window.</b> Customers may return most items within 30 days of delivery for a full refund. Items must be unused, in their original packaging, and accompanied by the receipt or order number.",
  "<b>2. Refund method and timing.</b> Approved refunds are issued to the original payment method within 5 to 7 business days after the returned item has been inspected at our warehouse. Customers receive an email once the refund is processed.",
  "<b>3. Shipping costs.</b> Original shipping fees are non-refundable unless the item arrived defective or was sent in error. Return shipping is free for defective items and paid by the customer in all other cases.",
  "<b>4. Non-refundable items.</b> Digital products, gift cards and personalised items cannot be refunded. Opened hygiene products cannot be returned.",
  "<b>5. Defective products.</b> Defective items are covered by a 90-day warranty and will be replaced or refunded at the customer's choice. Contact support with photos of the defect to start a claim."]),
 ("Security and Password Policy", [
  "<b>1. Password rules.</b> Passwords must contain at least 12 characters, including upper and lower case letters, a number and a symbol. Passwords must be changed every 90 days and must never be reused or shared with anyone, including IT staff.",
  "<b>2. Multi-factor authentication.</b> Multi-factor authentication is mandatory for email, the HR portal, the VPN and all production systems. Use only the company-approved password manager to store credentials.",
  "<b>3. After a password breach.</b> If you suspect your password has been compromised, change it immediately, then report the incident to the security team at security@company.example within 24 hours. Do not delete any suspicious emails or files, because they are needed as evidence. IT will end all active sessions and reset your credentials.",
  "<b>4. Lost or stolen devices.</b> Report a lost or stolen laptop or phone to the security team within 4 hours so that it can be remotely locked and wiped.",
  "<b>5. Phishing.</b> Report suspicious emails with the Report Phishing button. Never enter credentials on a page opened from an unexpected link."]),
]

def main(path: str = "data/sample_policy.pdf"):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(path, pagesize=A4, title="Sample Company Policies", author="Example Corp")
    story = []
    for i, (title, paras) in enumerate(PAGES):
        story += [Paragraph(title, styles["Title"]), Spacer(1, 12)]
        for p in paras:
            story += [Paragraph(p, styles["BodyText"]), Spacer(1, 8)]
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    doc.build(story); print("saved", path)

if __name__ == "__main__":
    main()
