package controller.user;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

@WebServlet("/auth/users/*")
public class UserServlet extends HttpServlet{
    protected void doGet(HttpServletRequest request, HttpServletResponse response){
        request.getPathInfo();
    }
}
