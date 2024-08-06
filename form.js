document.getElementById('sameAddressCheckbox').addEventListener('change', function () {
    var currentAddressInput = document.getElementById('currentAddress');
    currentAddressInput.disabled = this.checked;
    if (this.checked) {
        currentAddressInput.value = document.querySelector('[name="permanentAddress"]').value;
    } else {
        currentAddressInput.value = '';
    }
});

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
    clearErrors();

    var phone = document.forms['myForm']["fphone"].value;
    if (phone.length !== 10) {  
        seterror("phone", "*phone number must contain 10 numbers.");
        returnval = false;
    }
    var mail = document.forms['myForm']["fmail"].value;
    if (!mail.match(/^[A-Za-z\._\-0-9]*[@][A-Za-z]*[\.][a-z]{2,7}$/)) {
        seterror("mail", "*Please enter a valid email id.");
        returnval = false;
    }
    
    return returnval;
}
