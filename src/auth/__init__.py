"""
Authentication module for AURORA API.

Implements JWT-based authentication for secure API access.
"""

from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from pydantic import BaseModel
import os


# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-key-change-in-production")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Security scheme
security = HTTPBearer()


class TokenData(BaseModel):
    """Token payload data."""
    user_id: str
    username: Optional[str] = None
    exp: Optional[datetime] = None


class User(BaseModel):
    """User model."""
    user_id: str
    username: str
    email: Optional[str] = None
    is_active: bool = True


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Dictionary containing user data (must include 'sub' for user_id)
        expires_delta: Optional expiration time delta

    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> TokenData:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT token string

    Returns:
        TokenData with user information

    Raises:
        HTTPException: If token is invalid or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")

        if user_id is None:
            raise credentials_exception

        token_data = TokenData(
            user_id=user_id,
            username=payload.get("username"),
            exp=payload.get("exp")
        )

        return token_data

    except JWTError:
        raise credentials_exception


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """
    Dependency to get the current authenticated user from JWT token.

    Args:
        credentials: HTTP Bearer credentials from request header

    Returns:
        User object

    Raises:
        HTTPException: If authentication fails

    Usage:
        @app.get("/protected")
        async def protected_route(user: User = Depends(get_current_user)):
            return {"user_id": user.user_id}
    """
    token = credentials.credentials
    token_data = decode_access_token(token)

    # In a real application, you would fetch user from database
    # For now, we create a user from token data
    user = User(
        user_id=token_data.user_id,
        username=token_data.username or token_data.user_id,
        is_active=True
    )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )

    return user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """
    Optional authentication - allows both authenticated and anonymous access.

    Args:
        credentials: Optional HTTP Bearer credentials

    Returns:
        User object if authenticated, None otherwise
    """
    if credentials is None:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


# Helper function for testing/demo purposes
def create_demo_token(user_id: str = "demo_user") -> str:
    """
    Create a demo token for testing.

    Args:
        user_id: User ID to include in token

    Returns:
        JWT token string

    Usage:
        # For testing
        token = create_demo_token("user123")
        # Use in Authorization header: Bearer {token}
    """
    return create_access_token(
        data={"sub": user_id, "username": user_id},
        expires_delta=timedelta(hours=24)
    )


# Example: Generate a demo token when the module is run directly
if __name__ == "__main__":
    demo_token = create_demo_token("demo_user")
    print("Demo JWT Token:")
    print(demo_token)
    print("\nUse this in requests:")
    print(f'Authorization: Bearer {demo_token}')
    print("\nToken expires in 24 hours.")
