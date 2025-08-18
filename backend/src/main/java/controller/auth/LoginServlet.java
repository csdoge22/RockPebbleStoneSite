package controller.auth;

import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import io.jsonwebtoken.security.Keys;
import jakarta.servlet.ServletException;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.security.Key;
import java.util.Date;

@WebServlet("/api/auth/login")
public class LoginServlet extends HttpServlet {
    // Generate a secure key for HS256 (store this securely in production!)
    private static final Key SECRET_KEY = Keys.secretKeyFor(SignatureAlgorithm.HS256);
    private static final long EXPIRATION_TIME = 86400000; // 24 hours in milliseconds

    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        response.setContentType("application/json");
        String username = request.getParameter("username");
        String password = request.getParameter("password");
        
        if(username == null || password == null ||
           username.trim().isEmpty() || password.trim().isEmpty()){
            response.getWriter().write("{\"success\":false,\"error\":\"Missing credentials\"}");
            return;
        }

        String sql = "SELECT * FROM users WHERE username = ?";
        try(Connection conn = DriverManager.getConnection("jdbc:sqlite:/path/to/your/database.db")){
            // 1. First verify username exists
            PreparedStatement stmt = conn.prepareStatement(sql);
            stmt.setString(1, username);
            ResultSet rs = stmt.executeQuery();
            
            if(!rs.next()){
                response.getWriter().write("{\"success\":false,\"error\":\"Invalid username or password\"}");
                return;
            }
            
            // 2. Verify password (NEVER store plaintext passwords in production!)
            String storedPassword = rs.getString("password");
            if(!storedPassword.equals(password)) { // In real apps, use BCrypt!
                response.getWriter().write("{\"success\":false,\"error\":\"Invalid username or password\"}");
                return;
            }
            
            // 3. Create JWT token
            String token = Jwts.builder()
                .setSubject(username)
                .setIssuedAt(new Date())
                .setExpiration(new Date(System.currentTimeMillis() + EXPIRATION_TIME))
                .signWith(SECRET_KEY)
                .compact();
            
            // 4. Return success response with token
            response.getWriter().write("{\"success\":true,\"token\":\"" + token + "\",\"user\":{\"username\":\"" + username + "\"}}");
            
        } catch(SQLException e){
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
            response.getWriter().write("{\"success\":false,\"error\":\"Database error\"}");
        }
    }
}