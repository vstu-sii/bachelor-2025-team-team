CREATE TABLE `users` (
  `id` integer PRIMARY KEY NOT NULL AUTO_INCREMENT,
  `username` varchar(255),
  `email` varchar(255),
  `password` varchar(255),
  `role` varchar(255)
);

CREATE TABLE `vacancy` (
  `id` integer PRIMARY KEY NOT NULL AUTO_INCREMENT,
  `job_title` varchar(255),
  `education` varchar(255),
  `work_experience` integer,
  `desired_salary` varchar(255),
  `work_shedule` varchar(255),
  `work_format` varchar(255),
  `additional_requirements` varchar(255),
  `user_id` integer NOT NULL AUTO_INCREMENT
);

CREATE TABLE `resume` (
  `id` resume PRIMARY KEY NOT NULL AUTO_INCREMENT,
  `how_old` bool,
  `user_id` integer NOT NULL AUTO_INCREMENT
);

CREATE TABLE `score` (
  `id` score PRIMARY KEY NOT NULL AUTO_INCREMENT,
  `number_of_top` integer,
  `vacancy_id` integer NOT NULL AUTO_INCREMENT,
  `resume_id` integer NOT NULL AUTO_INCREMENT
);

ALTER TABLE `vacancy` ADD CONSTRAINT `user_vacancies` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`);

ALTER TABLE `resume` ADD CONSTRAINT `user_resume` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`);

ALTER TABLE `score` ADD CONSTRAINT `score_vacancy` FOREIGN KEY (`vacancy_id`) REFERENCES `vacancy` (`id`);

ALTER TABLE `score` ADD CONSTRAINT `score_resume` FOREIGN KEY (`resume_id`) REFERENCES `resume` (`id`);
