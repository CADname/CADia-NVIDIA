from __future__ import annotations

import os
import sys

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from standalonecad_web.auth import hash_password, normalize_email
from standalonecad_web.database import SessionLocal, init_database
from standalonecad_web.models import Project, User


def main() -> int:
    email = normalize_email(os.getenv("CADIA_JUDGE_EMAIL", "judge@cadia.co.kr"))
    password = os.getenv("CADIA_JUDGE_PASSWORD", "")
    if len(password) < 8:
        print("Set CADIA_JUDGE_PASSWORD to at least 8 characters.", file=sys.stderr)
        return 2
    display_name = os.getenv("CADIA_JUDGE_NAME", "Hackathon Judge")
    init_database()
    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.email == email))
        if user is None:
            user = User(email=email, display_name=display_name, password_hash=hash_password(password))
            db.add(user)
            db.flush()
        else:
            user.display_name = display_name
            user.password_hash = hash_password(password)
        if not db.scalar(select(Project).where(Project.owner_id == user.id, Project.name == "Judge Showcase Workspace")):
            db.add(Project(
                owner_id=user.id,
                name="Judge Showcase Workspace",
                description="Shared fallback workspace for Devpost judges. Prefer Launch CADia for isolated live AI testing.",
            ))
        try:
            db.commit()
        except IntegrityError as exc:
            db.rollback()
            raise exc
    print(f"Judge account ready: {email}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
