import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom"
import Register from "./Register";
import { describe, it, expect, beforeEach, vi } from "vitest";
import React from "react";

// Mock the register service
vi.mock("../services/RegisterService", () => ({
  register: vi.fn(),
}));

import { register } from "../services/RegisterService";

describe("Register form", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders all input fields and the submit button", () => {
    render(<Register />);
    expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /sign up/i })).toBeInTheDocument();
  });

  it("updates input values on change", () => {
    render(<Register />);
    const usernameInput = screen.getByLabelText(/username/i);
    fireEvent.change(usernameInput, { target: { value: "testuser", name: "username" } });
    expect(usernameInput.value).toBe("testuser");
  });

  it("calls register service and redirects on success", async () => {
    register.mockResolvedValueOnce({ success: true });
    delete window.location;
    window.location = { href: "" };

    render(<Register />);
    fireEvent.change(screen.getByLabelText(/username/i), { target: { value: "testuser", name: "username" } });
    fireEvent.change(screen.getByLabelText(/password/i), { target: { value: "testpass", name: "password" } });
    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: "test@example.com", name: "email" } });

    fireEvent.click(screen.getByRole("button", { name: /sign up/i }));

    await waitFor(() => {
      expect(register).toHaveBeenCalledWith({
        username: "testuser",
        password: "testpass",
        email: "test@example.com",
      });
      expect(window.location.href).toBe("/login");
    });
  });

  it("shows error message on registration failure", async () => {
    register.mockResolvedValueOnce({ success: false, error: "Username taken" });

    render(<Register />);
    fireEvent.change(screen.getByLabelText(/username/i), { target: { value: "testuser", name: "username" } });
    fireEvent.change(screen.getByLabelText(/password/i), { target: { value: "testpass", name: "password" } });
    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: "test@example.com", name: "email" } });

    fireEvent.click(screen.getByRole("button", { name: /sign up/i }));

    await waitFor(() => {
      expect(screen.getByText(/username taken/i)).toBeInTheDocument();
    });
  });
});