<!DOCTYPE html>
<!-- This page is just for testing backend endpoints -->
<html>
    <head>
        <title>Register Page</title>
    </head>
    <body>
        <h1>Register Account</h1>
        <form action="/backend/api/auth/register" method="post">
            <div>
                <label for="email">Email:</label>
                <!-- The name attribute is important for setting the name of the parameter for the backend -->
                <input type="text" id="email" name="email">
            </div>
            <div>
                <label for="name">Name:</label>
                <input type="text" id="name" name="name">
            </div>
            <div>
                <label for="password">Password:</label>
                <input type="password" id="password" name="password">
            </div>
            <div>
                <input type="submit" value="Register">
            </div>
        </form>
    </body>
</html>