User Service (MVP)

Endpoints:
- GET /health → { status: ok }
- POST /auth/register { email, password, full_name? } → 201 UserPublic
- POST /auth/login (OAuth2PasswordRequestForm: username=email, password) → TokenPair { access_token, refresh_token, token_type }
- POST /auth/refresh { refresh_token } → TokenPair
- GET /auth/me (Authorization: Bearer <access_token>) → UserPublic

Notes:
- JWTs: HS256, access expires in 30m, refresh in 7d. Configure via .env.
- Storage: In-memory dict for MVP. Replace with SQLAlchemy + SQLite/Postgres.
- Passwords: Hashed with bcrypt via passlib.
- Other services should validate only access tokens and never store refresh tokens. When expired, call /auth/refresh.
