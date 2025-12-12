const express = require('express');
const mysql = require('mysql2');
const path = require('path');
const { spawn } = require('child_process');
const session = require('express-session');
const multer = require('multer');
const keys = require(path.join(__dirname, "..", "keys.json"));

const app = express();
const PORT = 3000;

// Настройка multer для загрузки файлов в память (BLOB)
const storage = multer.memoryStorage();
const upload = multer({
    storage: storage,
    limits: {
        fileSize: 10 * 1024 * 1024 // Максимум 10MB
    },
    fileFilter: (req, file, cb) => {
        // Валидация расширения файла
        const allowedTypes = ['application/pdf'];
        const extname = path.extname(file.originalname).toLowerCase();
        
        if (allowedTypes.includes(file.mimetype) && extname === '.pdf') {
            cb(null, true);
        } else {
            cb(new Error('Разрешены только PDF файлы!'));
        }
    }
});

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
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

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

app.get('/upload', (req, res) => {
    res.sendFile(path.join(__dirname, "../frontend/resume.html"));
});

app.get('/ranking', (req, res) => {
    res.sendFile(path.join(__dirname, "../frontend/ranking.html"));
});

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

    const query = 'SELECT username, email FROM users WHERE id = ?';
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
// API МАРШРУТЫ - ПОЛЬЗОВАТЕЛИ
// ==========================================

// РЕГИСТРАЦИЯ
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
        const checkQuery = 'SELECT id FROM users WHERE username = ? OR email = ?';
        const [existingUsers] = await db.promise().query(checkQuery, [username, email]);

        if (existingUsers.length > 0) {
            return res.status(409).json({
                success: false,
                message: 'Пользователь с таким именем или email уже существует'
            });
        }

        const { hash, salt } = await hashPassword(password);

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
        const query = 'SELECT * FROM users WHERE email = ?';
        const [users] = await db.promise().query(query, [email]);

        if (users.length === 0) {
            return res.status(401).json({
                success: false,
                message: 'Неверный email или пароль'
            });
        }

        const user = users[0];

        const isValid = await verifyPassword(password, user.password_hash, user.password_salt);

        if (!isValid) {
            return res.status(401).json({
                success: false,
                message: 'Неверный email или пароль'
            });
        }

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
// API МАРШРУТЫ - ВАКАНСИИ
// ==========================================

// СОЗДАНИЕ ВАКАНСИИ
app.post('/api/vacancy/create', async (req, res) => {
    if (!req.session.userId) {
        return res.status(401).json({
            success: false,
            message: 'Необходима авторизация'
        });
    }

    const { jobTitle, education, workExperience, desiredSalary, workSchedule, workFormat, otherRequirements } = req.body;

    if (!jobTitle) {
        return res.status(400).json({
            success: false,
            message: 'Описание вакансии обязательно для заполнения'
        });
    }

    try {
        const insertQuery = `
            INSERT INTO vacancy 
            (job_title, education, work_experience, desired_salary, work_shedule, work_format, additional_requirements, user_id) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        `;

        const [result] = await db.promise().query(insertQuery, [
            jobTitle,
            education || null,
            workExperience || null,
            desiredSalary || null,
            workSchedule || null,
            workFormat || null,
            otherRequirements || null,
            req.session.userId
        ]);

        res.status(201).json({
            success: true,
            message: 'Вакансия успешно создана!',
            vacancyId: result.insertId
        });

    } catch (error) {
        console.error('Ошибка создания вакансии:', error);
        res.status(500).json({
            success: false,
            message: 'Ошибка сервера при создании вакансии'
        });
    }
});

// ПОЛУЧЕНИЕ ВАКАНСИЙ ПОЛЬЗОВАТЕЛЯ
app.get('/api/vacancy/user', async (req, res) => {
    if (!req.session.userId) {
        return res.status(401).json({
            success: false,
            message: 'Необходима авторизация'
        });
    }

    try {
        const query = 'SELECT * FROM vacancy WHERE user_id = ? ORDER BY id DESC';
        const [vacancies] = await db.promise().query(query, [req.session.userId]);

        res.status(200).json({
            success: true,
            vacancies: vacancies
        });

    } catch (error) {
        console.error('Ошибка получения вакансий:', error);
        res.status(500).json({
            success: false,
            message: 'Ошибка сервера'
        });
    }
});

// ==========================================
// API МАРШРУТЫ - РЕЗЮМЕ
// ==========================================

// ЗАГРУЗКА РЕЗЮМЕ (PDF в BLOB)
app.post('/api/resumes/upload', upload.array('resumes', 10), async (req, res) => {
    if (!req.session.userId) {
        return res.status(401).json({
            success: false,
            message: 'Необходима авторизация'
        });
    }

    if (!req.files || req.files.length === 0) {
        return res.status(400).json({
            success: false,
            message: 'Файлы не были загружены'
        });
    }

    try {
        const uploadedFiles = [];

        for (const file of req.files) {
            const insertQuery = 'INSERT INTO resume (file, user_id, how_old) VALUES (?, ?, 0)';
            const [result] = await db.promise().query(insertQuery, [file.buffer, req.session.userId]);

            uploadedFiles.push({
                id: result.insertId,
                filename: file.originalname,
                size: file.size
            });
        }

        res.status(201).json({
            success: true,
            message: `Успешно загружено ${uploadedFiles.length} файл(ов)`,
            files: uploadedFiles
        });

    } catch (error) {
        console.error('Ошибка загрузки резюме:', error);
        res.status(500).json({
            success: false,
            message: 'Ошибка сервера при загрузке файлов',
            error: error.message
        });
    }
});

// ПОЛУЧЕНИЕ СПИСКА РЕЗЮМЕ С РЕЙТИНГОМ
app.get('/api/resumes/ranked', async (req, res) => {
    if (!req.session.userId) {
        return res.status(401).json({
            success: false,
            message: 'Необходима авторизация'
        });
    }

    try {
        const query = `
            SELECT 
                r.id as resume_id,
                r.how_old,
                s.number_of_top,
                s.vacancy_id,
                LENGTH(r.file) as file_size
            FROM resume r
            LEFT JOIN score s ON r.id = s.resume_id
            WHERE r.user_id = ?
            ORDER BY s.number_of_top DESC
        `;
        const [resumes] = await db.promise().query(query, [req.session.userId]);

        res.status(200).json({
            success: true,
            resumes: resumes
        });

    } catch (error) {
        console.error('Ошибка получения списка резюме:', error);
        res.status(500).json({
            success: false,
            message: 'Ошибка сервера'
        });
    }
});

// СКАЧИВАНИЕ РЕЗЮМЕ ПО ID
app.get('/api/resumes/download/:id', async (req, res) => {
    if (!req.session.userId) {
        return res.status(401).json({
            success: false,
            message: 'Необходима авторизация'
        });
    }

    try {
        const query = 'SELECT file FROM resume WHERE id = ? AND user_id = ?';
        const [resumes] = await db.promise().query(query, [req.params.id, req.session.userId]);

        if (resumes.length === 0) {
            return res.status(404).json({
                success: false,
                message: 'Резюме не найдено'
            });
        }

        const resume = resumes[0];

        res.setHeader('Content-Type', 'application/pdf');
        res.setHeader('Content-Disposition', `attachment; filename="resume_${req.params.id}.pdf"`);
        res.send(resume.file);

    } catch (error) {
        console.error('Ошибка скачивания резюме:', error);
        res.status(500).json({
            success: false,
            message: 'Ошибка сервера'
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