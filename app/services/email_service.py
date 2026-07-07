from sqlalchemy.orm import Session

from app.models.auth import EmailOutbox


class EmailService:
    def send(self, db: Session, to_email: str, subject: str, body: str) -> None:
        outbox = EmailOutbox(to_email=to_email, subject=subject, body=body)
        db.add(outbox)
        db.flush()


email_service = EmailService()
