const express = require('express');
const mysql = require('mysql2');
const path = require('path');
const { spawn } = require('child_process');
const session = require('express-session');
const multer = require('multer');
const keys = require(path.join(__dirname, "..", "keys.json"));
const axios = require('axios');
const FormData = require('form-data');
const { Readable } = require('stream');

const app = express();
const PORT = 3000;
const AI_API_URL = 'http://localhost:8000';

/**
 * Конвертирует Buffer в Stream для отправки в FormData
 */
function bufferToStream(buffer) {
    const readable = new Readable();
    readable._read = () => {}; // Заглушка
    readable.push(buffer);
    readable.push(null);
    return readable;
}

/**
 * Форматирует данные вакансии в структурированный JSON
 * Возвращает объект с полями, где отсутствующие данные = null
 */
function formatVacancyDataToJSON(vacancy) {
    return {
        job_title: vacancy.job_title || null,
        education: vacancy.education || null,
        work_experience: vacancy.work_experience ? parseInt(vacancy.work_experience) : null,
        desired_salary: vacancy.desired_salary ? parseSalary(vacancy.desired_salary) : null,
        work_schedule: vacancy.work_shedule || null,
        work_format: vacancy.work_format || null,
        additional_requirements: vacancy.additional_requirements || null
    };
}

/**
 * Парсит зарплату из строки формата "От X до Y" или возвращает число
 */
function parseSalary(salaryString) {
    if (typeof salaryString === 'number') {
        return salaryString;
    }
    
    if (typeof salaryString === 'string') {
        // Формат: "От 50000 до 100000"
        const match = salaryString.match(/От\s*(\d+)/);
        if (match) {
            return parseInt(match[1]);
        }
        
        // Просто число в строке
        const numMatch = salaryString.match(/(\d+)/);
        if (numMatch) {
            return parseInt(numMatch[1]);
        }
    }
    
    return null;
}

/**
 * Форматирует данные вакансии в текст
 */
function formatVacancyData(vacancy) {
    const vacancyJSON = formatVacancyDataToJSON(vacancy);
    
    let vacancyText = `Vacancy Description:\n`;
    
    if (vacancyJSON.job_title) {
        vacancyText += `Position: ${vacancyJSON.job_title}\n`;
    }
    
    if (vacancyJSON.education) {
        vacancyText += `Education: ${vacancyJSON.education}\n`;
    }
    
    if (vacancyJSON.work_experience !== null) {
        vacancyText += `Work Experience: ${vacancyJSON.work_experience} years\n`;
    }
    
    if (vacancyJSON.desired_salary !== null) {
        vacancyText += `Desired Salary: ${vacancyJSON.desired_salary}\n`;
    }
    
    if (vacancyJSON.work_schedule) {
        vacancyText += `Work Schedule: ${vacancyJSON.work_schedule}\n`;
    }
    
    if (vacancyJSON.work_format) {
        vacancyText += `Work Format: ${vacancyJSON.work_format}\n`;
    }
    
    if (vacancyJSON.additional_requirements) {
        vacancyText += `Additional Requirements:\n${vacancyJSON.additional_requirements}\n`;
    }
    
    return vacancyText;
}

/**
 * Вызов AI API для оценки резюме
 */
async function evaluateResumeWithAI(resumeBuffer, vacancyData) {
    try {
        const formData = new FormData();

        const fileStream = bufferToStream(resumeBuffer);
        formData.append('file', fileStream, {
            filename: 'resume.pdf',
            contentType: 'application/pdf'
        });

        // Используем текстовый формат для оценки (как было)
        const vacancyText = formatVacancyData(vacancyData);
        formData.append('vacancy_data', vacancyText);

        const response = await axios.post(`${AI_API_URL}/match-vacancy`, formData, {
            headers: {
                ...formData.getHeaders()
            },
            timeout: 60000
        });

        console.log('AI API Response structure:', Object.keys(response.data));

        let overallScore = 0;
        let found = false;

        // Поиск overall_score в ответе
        if (response.data && response.data.matching_result && 
            response.data.matching_result.matching_results) {
            
            const matchingResults = response.data.matching_result.matching_results;
            if (typeof matchingResults.overall_score !== 'undefined') {
                overallScore = matchingResults.overall_score;
                console.log(`✅ Found overall_score in matching_result.matching_results: ${overallScore}`);
                found = true;
            }
        }
        
        if (!found && response.data && response.data.matching_result && 
            typeof response.data.matching_result.overall_score !== 'undefined') {
            
            overallScore = response.data.matching_result.overall_score;
            console.log(`✅ Found overall_score in matching_result: ${overallScore}`);
            found = true;
        }
        
        if (!found && response.data && response.data.matching_results && 
            typeof response.data.matching_results.overall_score !== 'undefined') {
            
            overallScore = response.data.matching_results.overall_score;
            console.log(`✅ Found overall_score in matching_results: ${overallScore}`);
            found = true;
        }
        
        if (!found && response.data && typeof response.data.overall_score !== 'undefined') {
            overallScore = response.data.overall_score;
            console.log(`✅ Found overall_score in root: ${overallScore}`);
            found = true;
        }

        if (found) {
            console.log(`✅ AI evaluation completed: ${overallScore} points`);
            return {
                success: true,
                score: overallScore,
                details: response.data.matching_result || response.data.matching_results || response.data
            };
        }

        console.warn('⚠️ Could not find overall_score in any expected location');
        console.warn('Full response:', JSON.stringify(response.data, null, 2));
        return {
            success: false,
            score: 0,
            error: 'Could not find overall_score in AI response'
        };

    } catch (error) {
        console.error('❌ AI API Error:', error.message);
        if (error.response) {
            console.error('Response status:', error.response.status);
            console.error('Response data:', error.response.data);
        }
        return {
            success: false,
            score: 0,
            error: error.message
        };
    }
}

/**
 * Генерация вопросов для собеседования через AI
 */
async function generateInterviewQuestions(resumeBuffer, vacancyData) {
    try {
        const formData = new FormData();

        // Конвертируем Buffer в Stream и добавляем как файл
        const fileStream = bufferToStream(resumeBuffer);
        formData.append('file', fileStream, {
            filename: 'resume.pdf',
            contentType: 'application/pdf'
        });

        // Форматируем данные вакансии в структурированный JSON
        const vacancyJSON = formatVacancyDataToJSON(vacancyData);
        
        // Отправляем как JSON строку
        formData.append('vacancy_requirements', JSON.stringify(vacancyJSON));
        formData.append('question_type', 'mixed');

        console.log('📤 Sending vacancy data to AI:', JSON.stringify(vacancyJSON, null, 2));

        // Отправляем запрос к AI API
        const response = await axios.post(`${AI_API_URL}/generate-interview-plan`, formData, {
            headers: {
                ...formData.getHeaders()
            },
            timeout: 90000 // 90 секунд таймаут для генерации вопросов
        });

        console.log('✅ Interview questions generated successfully');
        
        return {
            success: true,
            questions: response.data.interview_plan
        };

    } catch (error) {
        console.error('❌ AI API Error (interview questions):', error.message);
        if (error.response) {
            console.error('Response status:', error.response.status);
            console.error('Response data:', error.response.data);
        }
        return {
            success: false,
            error: error.message
        };
    }
}

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

app.get('/interview', (req, res) => {
    res.sendFile(path.join(__dirname, "../frontend/interview.html"));
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


// ПОЛУЧЕНИЕ ОДНОЙ ВАКАНСИИ ПО ID
app.get('/api/vacancy/:id', async (req, res) => {
    if (!req.session.userId) {
        return res.status(401).json({
            success: false,
            message: 'Необходима авторизация'
        });
    }

    try {
        const query = 'SELECT * FROM vacancy WHERE id = ? AND user_id = ?';
        const [vacancies] = await db.promise().query(query, [req.params.id, req.session.userId]);

        if (vacancies.length === 0) {
            return res.status(404).json({
                success: false,
                message: 'Вакансия не найдена'
            });
        }

        res.status(200).json({
            success: true,
            vacancy: vacancies[0]
        });

    } catch (error) {
        console.error('Ошибка получения вакансии:', error);
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


// ПОЛУЧЕНИЕ СПИСКА РЕАЛЬНЫХ РЕЗЮМЕ ПОЛЬЗОВАТЕЛЯ
app.get('/api/resumes/list', async (req, res) => {
    if (!req.session.userId) {
        return res.status(401).json({
            success: false,
            message: 'Необходима авторизация'
        });
    }

    try {
        const query = `
            SELECT 
                id,
                how_old,
                OCTET_LENGTH(file) AS file_size,
                DATE_FORMAT(created_at, '%Y-%m-%d %H:%i') AS upload_date
            FROM resume
            WHERE user_id = ?
              AND file IS NOT NULL
              AND OCTET_LENGTH(file) > 0
              AND created_at IS NOT NULL
            ORDER BY id DESC
        `;

        const [resumes] = await db.promise().query(query, [req.session.userId]);

        // 🔒 ЯВНАЯ ЛОГИЧЕСКАЯ ОБРАБОТКА
        if (!resumes || resumes.length === 0) {
            return res.status(200).json({
                success: true,
                resumes: []
            });
        }

        res.status(200).json({
            success: true,
            resumes
        });

    } catch (error) {
        console.error('Ошибка получения списка резюме:', error);
        res.status(500).json({
            success: false,
            message: 'Ошибка сервера'
        });
    }
});


// ОБРАБОТКА СУЩЕСТВУЮЩИХ РЕЗЮМЕ ДЛЯ ВАКАНСИИ (С AI)
app.post('/api/resumes/process', async (req, res) => {
    if (!req.session.userId) {
        return res.status(401).json({
            success: false,
            message: 'Необходима авторизация'
        });
    }

    const { resumeIds, vacancyId } = req.body;

    if (!resumeIds || !Array.isArray(resumeIds) || resumeIds.length === 0) {
        return res.status(400).json({
            success: false,
            message: 'Не выбраны резюме для обработки'
        });
    }

    if (!vacancyId) {
        return res.status(400).json({
            success: false,
            message: 'Не указана вакансия'
        });
    }

    try {
        // Проверяем, что все резюме принадлежат пользователю
        const checkQuery = 'SELECT id FROM resume WHERE id IN (?) AND user_id = ?';
        const [ownedResumes] = await db.promise().query(checkQuery, [resumeIds, req.session.userId]);

        if (ownedResumes.length !== resumeIds.length) {
            return res.status(403).json({
                success: false,
                message: 'Некоторые резюме не принадлежат текущему пользователю'
            });
        }

        // Получаем данные вакансии
        const vacancyQuery = 'SELECT * FROM vacancy WHERE id = ? AND user_id = ?';
        const [vacancies] = await db.promise().query(vacancyQuery, [vacancyId, req.session.userId]);

        if (vacancies.length === 0) {
            return res.status(403).json({
                success: false,
                message: 'Вакансия не найдена или не принадлежит пользователю'
            });
        }

        const vacancy = vacancies[0];
        
        // Счетчики для статистики
        let processedCount = 0;
        let successCount = 0;
        let errorCount = 0;
        let questionsGeneratedCount = 0;
        const results = [];

        console.log(`🚀 Starting AI evaluation for ${resumeIds.length} resume(s)...`);

        // Обрабатываем каждое резюме
        for (const resumeId of resumeIds) {
            try {
                // Получаем файл резюме из базы
                const resumeQuery = 'SELECT file FROM resume WHERE id = ?';
                const [resumes] = await db.promise().query(resumeQuery, [resumeId]);

                if (resumes.length === 0) {
                    console.warn(`⚠️ Resume ${resumeId} not found`);
                    errorCount++;
                    continue;
                }

                const resumeBuffer = resumes[0].file;

                // 1. ОЦЕНКА РЕЗЮМЕ через AI
                console.log(`🤖 Evaluating resume #${resumeId}...`);
                const aiResult = await evaluateResumeWithAI(resumeBuffer, vacancy);

                let finalScore = 0;

                if (aiResult.success) {
                    finalScore = aiResult.score;
                    
                    // Сохраняем оценку
                    const existingQuery = 'SELECT id FROM score WHERE resume_id = ? AND vacancy_id = ?';
                    const [existing] = await db.promise().query(existingQuery, [resumeId, vacancyId]);

                    if (existing.length === 0) {
                        const insertScore = 'INSERT INTO score (resume_id, vacancy_id, number_of_top) VALUES (?, ?, ?)';
                        await db.promise().query(insertScore, [resumeId, vacancyId, finalScore]);
                    } else {
                        const updateScore = 'UPDATE score SET number_of_top = ? WHERE resume_id = ? AND vacancy_id = ?';
                        await db.promise().query(updateScore, [finalScore, resumeId, vacancyId]);
                    }

                    successCount++;
                    console.log(`✅ Resume #${resumeId}: ${finalScore} points`);
                } else {
                    // Если AI не смог оценить, используем базовую оценку
                    finalScore = 0;
                    
                    const existingQuery = 'SELECT id FROM score WHERE resume_id = ? AND vacancy_id = ?';
                    const [existing] = await db.promise().query(existingQuery, [resumeId, vacancyId]);

                    if (existing.length === 0) {
                        const insertScore = 'INSERT INTO score (resume_id, vacancy_id, number_of_top) VALUES (?, ?, ?)';
                        await db.promise().query(insertScore, [resumeId, vacancyId, finalScore]);
                    } else {
                        const updateScore = 'UPDATE score SET number_of_top = ? WHERE resume_id = ? AND vacancy_id = ?';
                        await db.promise().query(updateScore, [finalScore, resumeId, vacancyId]);
                    }

                    errorCount++;
                    console.warn(`⚠️ Resume #${resumeId}: Using fallback score (${finalScore}) - ${aiResult.error}`);
                }

                // 2. ГЕНЕРАЦИЯ ВОПРОСОВ ДЛЯ СОБЕСЕДОВАНИЯ (только для успешно оцененных резюме)
                if (aiResult.success && finalScore > 0) {
                    console.log(`💡 Generating interview questions for resume #${resumeId}...`);
                    const questionsResult = await generateInterviewQuestions(resumeBuffer, vacancy);

                    if (questionsResult.success) {
                        // Сохраняем вопросы в базу данных
                        const questionsJSON = JSON.stringify(questionsResult.questions);
                        const updateQuestionsQuery = 'UPDATE resume SET questions = ? WHERE id = ?';
                        await db.promise().query(updateQuestionsQuery, [questionsJSON, resumeId]);
                        
                        questionsGeneratedCount++;
                        console.log(`✅ Interview questions generated for resume #${resumeId}`);
                    } else {
                        console.warn(`⚠️ Failed to generate questions for resume #${resumeId}: ${questionsResult.error}`);
                    }
                }

                results.push({
                    resumeId,
                    score: finalScore,
                    status: aiResult.success ? 'success' : 'fallback',
                    questionsGenerated: aiResult.success && finalScore > 0
                });

                processedCount++;

            } catch (error) {
                console.error(`❌ Error processing resume #${resumeId}:`, error);
                errorCount++;
                results.push({
                    resumeId,
                    status: 'error',
                    error: error.message
                });
            }
        }

        console.log(`📊 Processing complete: ${successCount} success, ${errorCount} errors, ${questionsGeneratedCount} questions generated`);

        // Формируем ответ
        const response = {
            success: true,
            message: `Обработано резюме: ${processedCount}`,
            processedCount,
            successCount,
            errorCount,
            questionsGeneratedCount,
            results
        };

        // Если были ошибки, добавляем предупреждение
        if (errorCount > 0) {
            response.warning = `${errorCount} резюме обработано с ошибками. Использованы средние оценки.`;
        }

        res.status(200).json(response);

    } catch (error) {
        console.error('❌ Fatal error in resume processing:', error);
        res.status(500).json({
            success: false,
            message: 'Ошибка сервера при обработке резюме',
            error: error.message
        });
    }
});

// Проверка доступности AI сервера
app.get('/api/ai/health', async (req, res) => {
    try {
        const response = await axios.get(`${AI_API_URL}/health`, {
            timeout: 5000
        });
        
        res.json({
            success: true,
            aiStatus: 'online',
            details: response.data
        });
    } catch (error) {
        res.json({
            success: false,
            aiStatus: 'offline',
            error: error.message
        });
    }
});

// ПОЛУЧЕНИЕ СПИСКА РЕЗЮМЕ С РЕЙТИНГОМ (С ФИЛЬТРАЦИЕЙ ПО ВАКАНСИИ)
app.get('/api/resumes/ranked', async (req, res) => {
    if (!req.session.userId) {
        return res.status(401).json({
            success: false,
            message: 'Необходима авторизация'
        });
    }

    const { vacancyId } = req.query;

    if (!vacancyId) {
        return res.status(400).json({
            success: false,
            message: 'Не указана вакансия'
        });
    }

    try {
        // Проверяем, что вакансия принадлежит пользователю
        const vacancyCheck = 'SELECT id FROM vacancy WHERE id = ? AND user_id = ?';
        const [vacancies] = await db.promise().query(vacancyCheck, [vacancyId, req.session.userId]);

        if (vacancies.length === 0) {
            return res.status(403).json({
                success: false,
                message: 'Вакансия не найдена или не принадлежит пользователю'
            });
        }

        // Получаем резюме только для указанной вакансии
        const query = `
            SELECT 
                r.id as resume_id,
                r.how_old,
                s.number_of_top,
                s.vacancy_id,
                LENGTH(r.file) as file_size
            FROM resume r
            INNER JOIN score s ON r.id = s.resume_id
            WHERE r.user_id = ? AND s.vacancy_id = ?
            ORDER BY s.number_of_top DESC, r.id ASC
        `;
        const [resumes] = await db.promise().query(query, [req.session.userId, vacancyId]);

        res.status(200).json({
            success: true,
            resumes: resumes,
            vacancyId: vacancyId
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

// ПОЛУЧЕНИЕ ВОПРОСОВ ДЛЯ СОБЕСЕДОВАНИЯ
app.get('/api/interviews/questions', async (req, res) => {
    if (!req.session.userId) {
        return res.status(401).json({
            success: false,
            message: 'Необходима авторизация'
        });
    }

    const { vacancyId } = req.query;

    if (!vacancyId) {
        return res.status(400).json({
            success: false,
            message: 'Не указана вакансия'
        });
    }

    try {
        // Проверяем, что вакансия принадлежит пользователю
        const vacancyCheck = 'SELECT id FROM vacancy WHERE id = ? AND user_id = ?';
        const [vacancies] = await db.promise().query(vacancyCheck, [vacancyId, req.session.userId]);

        if (vacancies.length === 0) {
            return res.status(403).json({
                success: false,
                message: 'Вакансия не найдена или не принадлежит пользователю'
            });
        }

        // Получаем резюме с вопросами для указанной вакансии
        const query = `
            SELECT 
                r.id as resume_id,
                r.questions,
                s.number_of_top as score,
                s.vacancy_id
            FROM resume r
            INNER JOIN score s ON r.id = s.resume_id
            WHERE r.user_id = ? 
              AND s.vacancy_id = ?
              AND r.questions IS NOT NULL
            ORDER BY s.number_of_top DESC
        `;
        
        const [interviews] = await db.promise().query(query, [req.session.userId, vacancyId]);

        res.status(200).json({
            success: true,
            interviews: interviews,
            vacancyId: vacancyId
        });

    } catch (error) {
        console.error('Ошибка получения вопросов:', error);
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