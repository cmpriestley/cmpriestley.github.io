# Personal Website

This is my personal website hosted on GitHub Pages.

## Structure

- `index.html` - Homepage with personal information and profile picture
- `notes.html` - Lecture notes page organized by course
- `style.css` - Stylesheet for the website (dark theme)
- `gr-notes.html` - Transcribed General Relativity notes
- `astrostatistics-notes.html` - Transcribed Astrostatistics notes
- `notes/` - Folder containing PDF lecture notes organized by course
  - `general-relativity/` - General Relativity course notes (PDF)
  - `astrostatistics/` - Astrostatistics course notes (PDF)

## Features

- Dark theme with black background
- Profile picture placeholder with rounded corners
- Handwritten notes with Gemini-transcribed versions using MCP Handley Lab Toolkit

## Adding New Lecture Notes

1. Create a new folder in `notes/` for the course (e.g., `notes/course-name/`)
2. Add your PDF files to the course folder
3. Generate transcribed notes using Gemini and save as `course-name-notes.html`
4. Edit `notes.html` to add links to both the PDF and transcribed HTML files
5. Commit and push the changes to GitHub

## Customization

Feel free to:
- Edit `index.html` to update personal information
- Add your profile picture by replacing the comment in the `.profile-picture` div
- Add new course sections in `notes.html`
- Create new folders in `notes/` for additional courses
- Modify `style.css` to change colors and appearance
