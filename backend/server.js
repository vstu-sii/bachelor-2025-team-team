const express = require('express');
const mysql = require('mysql2');
const path = require('path');
const { spawn } = require('child_process');
const session = require('express-session');
const keys = require(path.join(__dirname, "..", "keys.json"));

const app = express();
const PORT = 3000;

app.use(session({
    secret: keys.secret,
    resave: false,
    saveUninitialized: false,
    cookie: {
        secure: false,
        httpOnly: true,
        maxAge: 24 * 60 * 60 * 1000 // 24 часа
    }
}));

app.use(express.static(path.join(__dirname, '../frontend')));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Подключение к базе данных
const db = mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password: keys.bd_pass,
    database: 'SII',
    port: 3307
});

db.connect((err) => {
    if (err) {
        console.error('Ошибка подключения к базе данных', err.stack);
        return;
    }
    console.log('Успешное подключение к базе данных.');
});

// ==========================================
// ФУНКЦИИ ХЭШИРОВАНИЯ
// ==========================================

function hashPassword(password) {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python', [
            path.join(__dirname, 'hash.py'),
            password
        ]);

        let result = '';
        let error = '';

        pythonProcess.stdout.on('data', (data) => {
            result += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            error += data.toString();
        });

        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(error || 'Python script failed'));
            } else {
                try {
                    const parts = result.split(/\r?\n/).map(s => s.trim());
                    const hash = parts[0];
                    const salt = parts[1];
                    resolve({ hash, salt });
                } catch (e) {
                    reject(new Error('Invalid response from hasher'));
                }
            }
        });
    });
}

function verifyPassword(password, hash, salt) {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python', [
            path.join(__dirname, 'hash.py'),
            'verify',
            password,
            hash,
            salt
        ]);

        let result = '';

        pythonProcess.stdout.on('data', (data) => {
            result += data.toString();
        });

        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                reject(new Error('Verification failed'));
            } else {
                resolve(result.trim().toLowerCase() === 'true');
            }
        });
    });
}

// ==========================================
// МАРШРУТЫ ДЛЯ СТРАНИЦ
// ==========================================

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/index.html'));
});

app.get('/authorization', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/authorization.html'));
});

app.get('/registration', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/registration.html'));
});

app.get('/vacancy', (req, res) => {
    res.sendFile(path.join(__dirname, "../frontend/vacancy.html"));
});

// ИСПРАВЛЕННЫЙ LOGOUT
app.get('/logout', (req, res) => {
    req.session.destroy((err) => {
        if (err) return res.status(500).send('Ошибка при выходе');
        res.clearCookie('connect.sid');
        res.redirect('/');
    });
});

app.get('/api/user', (req, res) => {
    if (!req.session.userId) {
        return res.status(401).json({ success: false, message: 'Пользователь не авторизован' });
    }

    const query = 'SELECT login, email FROM user WHERE id = ?';
    db.query(query, [req.session.userId], (err, results) => {
        if (err) {
            console.error('Ошибка при выполнении запроса:', err);
            return res.status(500).json({ success: false, message: 'Ошибка сервера' });
        }

        if (results.length > 0) {
            res.json({ success: true, user: results[0] });
        } else {
            res.status(404).json({ success: false, message: 'Пользователь не найден' });
        }
    });
});

// ==========================================
// API МАРШРУТЫ
// ==========================================

// РЕГИСТРАЦИЯ (ИСПРАВЛЕНО ДЛЯ MySQL)
app.post('/api/signup', async (req, res) => {
    const { username, email, password } = req.body;

    if (!username || !email || !password) {
        return res.status(400).json({
            success: false,
            message: 'Все поля обязательны для заполнения'
        });
    }

    if (username.length < 3) {
        return res.status(400).json({
            success: false,
            message: 'Имя пользователя должно содержать минимум 3 символа'
        });
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
        return res.status(400).json({
            success: false,
            message: 'Неверный формат email'
        });
    }

    if (password.length < 8) {
        return res.status(400).json({
            success: false,
            message: 'Пароль должен содержать минимум 8 символов'
        });
    }

    try {
        // Проверка существования пользователя (MYSQL)
        const checkQuery = 'SELECT id FROM users WHERE username = ? OR email = ?';
        const [existingUsers] = await db.promise().query(checkQuery, [username, email]);

        if (existingUsers.length > 0) {
            return res.status(409).json({
                success: false,
                message: 'Пользователь с таким именем или email уже существует'
            });
        }

        // Хэширование пароля
        const { hash, salt } = await hashPassword(password);

        // Вставка нового пользователя (MYSQL)
        const insertQuery = `INSERT INTO users (username, email, password_hash, password_salt) VALUES (?, ?, ?, ?)`;
        const [result] = await db.promise().query(insertQuery, [username, email, hash, salt]);

        res.status(201).json({
            success: true,
            message: 'Регистрация успешна!',
            userId: result.insertId
        });

    } catch (error) {
        console.error('Ошибка регистрации:', error);
        res.status(500).json({
            success: false,
            message: 'Ошибка сервера при регистрации'
        });
    }
});

// АВТОРИЗАЦИЯ (ИСПРАВЛЕНО ДЛЯ MySQL + СЕССИЯ)
app.post('/api/signin', async (req, res) => {
    const { email, password } = req.body;

    if (!email || !password) {
        return res.status(400).json({
            success: false,
            message: 'Email и пароль обязательны'
        });
    }

    try {
        // Получение пользователя из БД (MYSQL)
        const query = 'SELECT * FROM users WHERE email = ?';
        const [users] = await db.promise().query(query, [email]);

        if (users.length === 0) {
            return res.status(401).json({
                success: false,
                message: 'Неверный email или пароль'
            });
        }

        const user = users[0];

        // Проверка пароля
        const isValid = await verifyPassword(password, user.password_hash, user.password_salt);

        if (!isValid) {
            return res.status(401).json({
                success: false,
                message: 'Неверный email или пароль'
            });
        }

        // СОЗДАНИЕ СЕССИИ
        req.session.userId = user.id;
        req.session.username = user.username;
        req.session.email = user.email;

        res.status(200).json({
            success: true,
            message: 'Вход выполнен успешно!',
            user: {
                id: user.id,
                username: user.username,
                email: user.email
            }
        });

    } catch (error) {
        console.error('Ошибка входа:', error);
        res.status(500).json({
            success: false,
            message: 'Ошибка сервера при входе'
        });
    }
});

// ==========================================
// ЗАПУСК СЕРВЕРА
// ==========================================

app.listen(PORT, () => {
    console.log(`✓ Сервер запущен на http://localhost:${PORT}`);
});

process.on('SIGINT', () => {
    db.end((err) => {
        if (err) {
            console.error('Ошибка закрытия БД:', err.message);
        }
        console.log('\n✓ База данных закрыта');
        process.exit(0);
    });
});