package controller.auth;

import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;

import jakarta.servlet.ServletException;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

@WebServlet("/api/auth/register")
public class RegisterServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        response.setContentType("application/json");
        String email = request.getParameter("email");
        String name = request.getParameter("username");
        String password = request.getParameter("password");

        if(email == null || name == null || password == null ||
           email.trim().isEmpty() || name.trim().isEmpty() || password.trim().isEmpty()){
            response.getWriter().write("{\"success\":false,\"error\":\"You are missing an input\"}");
            return;
        }
        String sql = "INSERT INTO users (email, username, password) VALUES (?, ?, ?)";
        try (Connection conn = DriverManager.getConnection("jdbc:sqlite:/Users/sonitambashta/Desktop/PYTHON/RockPebbleStoneSite/backend/database/sandsolutionsdb.db")){
            PreparedStatement pstmt = conn.prepareStatement(sql);
            pstmt.setString(1,email);
            pstmt.setString(2,name);
            pstmt.setString(3,password);
            pstmt.executeUpdate();
            response.getWriter().write("{\"success\":true}");
        } catch(SQLException e){
            response.getWriter().write("{\"success\":false,\"error\":\"Database error: " + e.getMessage().replace("\"", "\\\"") + "\"}");
        }
    }
}
