

import dao.CreateDatabase;
import jakarta.servlet.annotation.WebListener;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.ServletContextEvent;
import jakarta.servlet.ServletContextListener;
import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

@WebListener
/** Initializes the database connection before servlets run */
public class DatabaseInitListener implements ServletContextListener {
    public void contextInitialized(ServletContextEvent sce) {
        CreateDatabase.connect();
    }
}
