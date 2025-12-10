const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const { spawn } = require('child_process');

const app = express();
const PORT = 3000;

// Разрешаем отдачу статических файлов (HTML, CSS)
app.use(express.static(path.join(__dirname, '../frontend')));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Подключение к базе данных
const db = new sqlite3.Database('./database/database.db', (err) => {
    if (err) {
        console.error('Ошибка подключения к БД:', err.message);
    } else {
        console.log('✓ Подключено к базе данных');
    }
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
                    const [hash, salt] = result.trim().split('\n');
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
                resolve(result.trim() === 'True');
            }
        });
    });
}

// ==========================================
// МАРШРУТЫ ДЛЯ СТРАНИЦ
// ==========================================

// Главная страница
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/index.html'));
});

// Переход на страницу авторизации
app.get('/authorization', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/authorization.html'));
});

// Переход на страницу регистрации
app.get('/registration', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/registration.html'));
});

// Переход на страницу с вакансиями
app.get('/vacancy', (req, res) => {
    res.sendFile(path.join(__dirname, "../frontend/vacancy.html"));
});

// ==========================================
// API МАРШРУТЫ
// ==========================================

// РЕГИСТРАЦИЯ
app.post('/api/signup', async (req, res) => {
    const { username, email, password } = req.body;

    // Валидация входных данных
    if (!username || !email || !password) {
        return res.status(400).json({
            success: false,
            message: 'Все поля обязательны для заполнения'
        });
    }

    // Проверка длины username
    if (username.length < 3) {
        return res.status(400).json({
            success: false,
            message: 'Имя пользователя должно содержать минимум 3 символа'
        });
    }

    // Проверка email
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
        return res.status(400).json({
            success: false,
            message: 'Неверный формат email'
        });
    }

    // Проверка длины пароля
    if (password.length < 8) {
        return res.status(400).json({
            success: false,
            message: 'Пароль должен содержать минимум 8 символов'
        });
    }

    try {
        // Проверка существования пользователя
        const existingUser = await new Promise((resolve, reject) => {
            db.get(
                'SELECT id FROM users WHERE username = ? OR email = ?',
                [username, email],
                (err, row) => {
                    if (err) reject(err);
                    else resolve(row);
                }
            );
        });

        if (existingUser) {
            return res.status(409).json({
                success: false,
                message: 'Пользователь с таким именем или email уже существует'
            });
        }

        // Хэширование пароля через Python
        const { hash, salt } = await hashPassword(password);

        // Вставка нового пользователя в БД
        const result = await new Promise((resolve, reject) => {
            db.run(
                `INSERT INTO users (username, email, password_hash, password_salt) 
                 VALUES (?, ?, ?, ?)`,
                [username, email, hash, salt],
                function(err) {
                    if (err) reject(err);
                    else resolve(this.lastID);
                }
            );
        });

        // Успешная регистрация
        res.status(201).json({
            success: true,
            message: 'Регистрация успешна!',
            userId: result
        });

    } catch (error) {
        console.error('Ошибка регистрации:', error);
        res.status(500).json({
            success: false,
            message: 'Ошибка сервера при регистрации'
        });
    }
});

// АВТОРИЗАЦИЯ
app.post('/api/signin', async (req, res) => {
    const { email, password } = req.body;

    if (!email || !password) {
        return res.status(400).json({
            success: false,
            message: 'Email и пароль обязательны'
        });
    }

    try {
        // Получение пользователя из БД
        const user = await new Promise((resolve, reject) => {
            db.get(
                'SELECT * FROM users WHERE email = ?',
                [email],
                (err, row) => {
                    if (err) reject(err);
                    else resolve(row);
                }
            );
        });

        if (!user) {
            return res.status(401).json({
                success: false,
                message: 'Неверный email или пароль'
            });
        }

        // Проверка пароля через Python
        const isValid = await verifyPassword(password, user.password_hash, user.password_salt);

        if (!isValid) {
            return res.status(401).json({
                success: false,
                message: 'Неверный email или пароль'
            });
        }

        // Успешный вход
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

// Корректное завершение при выходе
process.on('SIGINT', () => {
    db.close((err) => {
        if (err) {
            console.error('Ошибка закрытия БД:', err.message);
        }
        console.log('\n✓ База данных закрыта');
        process.exit(0);
    });
});