# 🌐 Rock-Pebble-Sand API Endpoints

This document outlines the RESTful API endpoints for the Rock-Pebble-Sand application. The backend is implemented using Java Servlets, with frontend forms or JS-based calls.

---

## 📁 Authentication Endpoints

### 🔐 POST `/api/auth/login`
**Purpose**: Log in an existing user.

- **Request Parameters** (form or JSON):
  - `username`: String
  - `password`: String

- **Response**:
  - `200 OK` with session created and user JSON
  - `401 Unauthorized` if login fails

---

### 📝 POST `/api/auth/register`
**Purpose**: Register a new user.

- **Request Parameters**:
  - `username`: String
  - `password`: String
  - `email`: String (optional)

- **Response**:
  - `201 Created` if successful
  - `400 Bad Request` if user already exists

---

### 🔓 POST `/api/auth/logout`
**Purpose**: Invalidate user session.

- **Response**:
  - `200 OK` with message
  - Invalidates `HttpSession`

---

## 👤 User Account Endpoints

### 📄 GET `/api/user/me`
**Purpose**: Get the current logged-in user session data.

- **Response**:
  - `200 OK` with user object
  - `401 Unauthorized` if not logged in

---

### ✏️ PUT `/api/user/update`
**Purpose**: Update user profile (e.g., password or email).

- **Request Parameters**:
  - `email`: String (optional)
  - `password`: String (optional)

- **Response**:
  - `200 OK` if updated
  - `400 Bad Request` if validation fails

---

### ❌ DELETE `/api/user/delete`
**Purpose**: Delete user account.

- **Response**:
  - `200 OK` if deleted
  - `401 Unauthorized` if not logged in

---


## 🧩 Technical Notes

- All endpoints starting with `/api/` should be routed to corresponding `*.java` servlets like `AuthServlet`, `UserServlet`, etc.
- Sessions are managed via `HttpSession`.
- JSON output is recommended for frontend compatibility.
- SQLite is used as default for dev, but migration to cloud DBs is supported.

---

## 🧰 Servlet Mappings (Example)

| Endpoint                | Servlet Class       | Description         |
|------------------------|---------------------|---------------------|
| `/api/auth/login`      | `AuthServlet`       | Handles login       |
| `/api/auth/register`   | `AuthServlet`       | Handles registration|
| `/api/auth/logout`     | `AuthServlet`       | Handles logout      |
| `/api/user/*`          | `UserServlet`       | User CRUD operations|

Make sure you configure these in your `web.xml` or via annotations like `@WebServlet("/api/auth/*")`.

---

