package controller.auth;

import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

import jakarta.servlet.ServletException;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

@WebServlet("/api/auth/login")
public class LoginServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        response.setContentType("application/json");
        String username = request.getParameter("username");
        String password = request.getParameter("password");
        System.out.println("Login attempt with username: " + username);
        System.out.println("Login attempt with password: " + password);
        
        if(username == null || password == null ||
           username.trim().isEmpty() || password.trim().isEmpty()){
            response.getWriter().write("{\"success\":false,\"error\":\"You are missing an input\"}");
            return;
        }
        String sql = "SELECT * FROM users WHERE username = ? AND password = ?";
        try(Connection conn = DriverManager.getConnection("jdbc:sqlite:/Users/sonitambashta/Desktop/PYTHON/RockPebbleStoneSite/backend/database/sandsolutionsdb.db")){
            PreparedStatement stmt = conn.prepareStatement(sql);
            stmt.setString(1, username);
            stmt.setString(2, password);
            ResultSet rs = stmt.executeQuery();
            if(rs.next()){
                response.getWriter().write("{\"success\":true}");
            } else {
                response.getWriter().write("{\"success\":false,\"error\":\"Invalid username or password\"}");
            }
        } catch(SQLException e){
            response.getWriter().write("{\"success\":false,\"error\":\"Database error: " + e.getMessage().replace("\"", "\\\"") + "\"}");
        }
    }
}
