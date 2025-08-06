package dao;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;

public class CreateDatabase {
    public static void connect() {
        // Assuming the database URL is stored in a file, but will move the location before deployment
        String url = "jdbc:sqlite:/Users/sonitambashta/Desktop/PYTHON/RockPebbleStoneSite/backend/database/sandsolutionsdb.db";
        if(url.isEmpty()){
            System.out.println("Database URL is empty. Please check the file.");
            return;
        }
        try {
            Class.forName("org.sqlite.JDBC"); // <-- Move this line before getConnection
            try (Connection conn = DriverManager.getConnection(url)) {
                if (conn != null) {
                    System.out.println("Connection to SQLite has been established.");
                    createTables(conn);
                }
            }
        } catch (SQLException e) {
            System.out.println(e.getMessage());
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    public static void createTables(Connection conn){
        System.out.println("Creating tables");
        String sql = "CREATE TABLE IF NOT EXISTS users (\n"
                + " id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
                + " username TEXT NOT NULL UNIQUE,\n"
                + " password TEXT NOT NULL,\n"
                + " email TEXT NOT NULL UNIQUE\n"
                + ");";
        String sql2 = "CREATE TABLE IF NOT EXISTS tasks (\n"
                + " id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
                + " user_id INTEGER NOT NULL,\n"
                + " description TEXT,\n"
                + " category REAL NOT NULL\n"
                + ");";

        try (PreparedStatement stmt = conn.prepareStatement(sql)) {
            stmt.execute();
            System.out.println("Tables created successfully");
        } catch (SQLException e) {
            System.out.println(e.getMessage());
        }
        try (PreparedStatement stmt = conn.prepareStatement(sql2)) {
            stmt.execute();
            System.out.println("Tasks table created successfully");
        } catch (SQLException e) {
            System.out.println(e.getMessage());
        }
    }
}