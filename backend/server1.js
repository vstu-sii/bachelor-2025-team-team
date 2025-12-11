const express = require('express');
const mysql = require('mysql2');
const path = require('path');
const bodyParser = require('body-parser');
const session = require('express-session');

const app = express();
const port = 3000;

const db = mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password: 'root',
    database: 'SII'
});

db.connect((err) => {
    if (err) {
        console.error('Ошибка подключения к бае данных');
        return;
    }
    console.log('Подключение к базе данных успешно.');
});

