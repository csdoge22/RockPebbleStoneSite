package controller.auth;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.ServletException;
import java.io.*;

import dao.CreateDatabase;

@WebServlet("/api/auth/*")
public class AuthServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        CreateDatabase.connect();
        String path = request.getPathInfo(); // Gets part after /api/auth

        response.setContentType("application/json");

        switch (path) {
            case "/login":
                handleLogin(request, response);
                break;
            case "/register":
                handleRegister(request, response);
                break;
            case "/logout":
                handleLogout(request, response);
                break;
            default:
                response.setStatus(HttpServletResponse.SC_NOT_FOUND);
                response.getWriter().write("{\"error\": \"Endpoint not found\"}");
        }
    }

    private void handleLogin(HttpServletRequest request, HttpServletResponse response) throws IOException {
        // retrieve request body

        StringBuilder sb = new StringBuilder();
        String line;
        try (BufferedReader reader = request.getReader()) {
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
        }
        String requestBody = sb.toString();
        System.out.println(requestBody);
        response.getWriter().write("{\"message\": \"Login successful\"}");
    }

    private void handleRegister(HttpServletRequest request, HttpServletResponse response) throws IOException {
        StringBuilder sb = new StringBuilder();
        String line;
        try (BufferedReader reader = request.getReader()) {
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
        }
        String requestBody = sb.toString();
        System.out.println(requestBody);
        response.getWriter().write("{\"message\": \"Registration successful\"}");
    }

    private void handleLogout(HttpServletRequest request, HttpServletResponse response) throws IOException {
        // Invalidate the session
        request.getSession().invalidate();
        response.getWriter().write("{\"message\": \"Logged out successfully\"}");
    }
}
