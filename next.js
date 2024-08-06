function togglePasswordVisibility() {
      var passwordInput = document.getElementById("passwordInput");
      var showPasswordCheckbox = document.getElementById("showPasswordCheckbox");
  
      if (showPasswordCheckbox.checked) {
        passwordInput.type = "text";
      } else {
        passwordInput.type = "password";
      }
    }
    function clearErrors() {
      errors = document.getElementsByClassName('formerror');
      for (let item of errors) {
        item.innerHTML = "";
      }
    } 
    function seterror(id, error) {
      element = document.getElementById(id);
      element.getElementsByClassName('formerror')[0].innerHTML = error;
      element.getElementsByClassName('formerror')[0].style.color = 'red';
    } 
    function validateForm() {
      var returnval = true;
      var errorMessages = [];
      clearErrors();
  
      var name = document.forms['myForm']["fname"].value;
      if (name.length < 5) {
        seterror("name", "*User name is too short.");
        returnval = false;
      }
      if (name.length == 0) {
        seterror("name", "*This column cannot be empty.");
        returnval = false;
      }
      var password = document.forms['myForm']["fpass"].value;
      if (password.length < 8) {
        errorMessages.push("*Password should contain at least 8 characters.");
        returnval = false;
      }
      if (!/\d/.test(password)) {
        errorMessages.push("*Password should contain at least one number.");
        returnval = false;
      }
      if (!/[A-Z]/.test(password)) {
        errorMessages.push("*Password should contain at least one uppercase letter.");
        returnval = false;
      }
      if (!returnval) {
        seterror("pass", errorMessages.join("<br>"));
      }
      var cpassword = document.forms['myForm']["fcpass"].value;
      if (cpassword != password) {
        seterror("cpass", "*Password do not match.");
        returnval = false;
      }

      return returnval;
  }
  document.getElementById('registrationForm').addEventListener('submit', function (event) {
    event.preventDefault();
    if (validateForm()) {
        swal("Good job!", "Registration Completed Successfully!", "success",).then(() => {
            window.location.href = 'index.html';
        });
    }
});