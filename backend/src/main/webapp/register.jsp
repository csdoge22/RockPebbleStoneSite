<!DOCTYPE html>
<!-- This page is just for testing backend endpoints -->
<html>
    <head>
        <title>Register Page</title>
    </head>
    <body>
        <h1>Register Account</h1>
        <form action="/auth/register" method="post">
            <div>
                <label for="email">Email:</label>
                <!-- The name attribute is important for setting the name of the parameter for the backend -->
                <input type="text" id="email" name="email" required>
            </div>
            <div>
                <label for="name">Name:</label>
                <input type="text" id="name" name="name" required>
            </div>
            <div>
                <label for="password">Password:</label>
                <input type="password" id="password" name="password" required>
            </div>
            <div>
                <input type="submit" value="Register">
            </div>
        </form>
    </body>
</html>