use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use image::{imageops, GrayImage, Luma, Rgb, RgbImage};
use imageproc::contrast::{equalize_histogram, otsu_level};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::filter::median_filter;
use imageproc::rect::Rect;
use imageproc::region_labelling::{connected_components, Connectivity};

fn make_test_image(path: &Path) {
    let (w, h) = (1200u32, 1600u32);
    let mut img = RgbImage::from_pixel(w, h, Rgb([255u8, 255, 255]));
    let black = Rgb([0u8, 0, 0]);
    for t in 0..8u32 {
        for x in 80..1120 { img.put_pixel(x, 80 + t, black); img.put_pixel(x, 1512 + t, black); }
        for y in 80..1520 { img.put_pixel(80 + t, y, black); img.put_pixel(1112 + t, y, black); }
    }
    let mut y = 200;
    while y < 1400 {
        for yy in y..y + 12 { for x in 140..1000 { img.put_pixel(x, yy, black); } }
        y += 80;
    }
    img.save(path).expect("failed to save test image");
}

fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

fn solve_homography(src: [(f32, f32); 4], dst: [(f32, f32); 4]) -> Option<[f32; 9]> {
    let mut a = [[0.0f32; 8]; 8];
    let mut b = [0.0f32; 8];
    for i in 0..4 {
        let (x, y) = dst[i];
        let (u, v) = src[i];
        a[2*i][0] = x; a[2*i][1] = y; a[2*i][2] = 1.0; a[2*i][6] = -x*u; a[2*i][7] = -y*u; b[2*i] = u;
        a[2*i+1][3] = x; a[2*i+1][4] = y; a[2*i+1][5] = 1.0; a[2*i+1][6] = -x*v; a[2*i+1][7] = -y*v; b[2*i+1] = v;
    }
    for i in 0..8 {
        let mut max_row = i;
        for k in i+1..8 { if a[k][i].abs() > a[max_row][i].abs() { max_row = k; } }
        a.swap(i, max_row); b.swap(i, max_row);
        if a[i][i].abs() < 1e-6 { return None; }
        for k in i+1..8 {
            let factor = a[k][i] / a[i][i];
            for j in i..8 { a[k][j] -= factor * a[i][j]; }
            b[k] -= factor * b[i];
        }
    }
    let mut h = [0.0f32; 8];
    for i in (0..8).rev() {
        let mut sum = b[i];
        for j in i+1..8 { sum -= a[i][j] * h[j]; }
        h[i] = sum / a[i][i];
    }
    Some([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0])
}

fn warp_perspective(img: &GrayImage, h: &[f32; 9], out_w: u32, out_h: u32) -> GrayImage {
    let mut out = GrayImage::from_pixel(out_w, out_h, Luma([255]));
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    for y in 0..out_h {
        for x in 0..out_w {
            let xf = x as f32; let yf = y as f32;
            let w = h[6]*xf + h[7]*yf + h[8];
            if w.abs() < 1e-6 { continue; }
            let u = (h[0]*xf + h[1]*yf + h[2]) / w;
            let v = (h[3]*xf + h[4]*yf + h[5]) / w;
            let u0 = u.floor() as i32; let v0 = v.floor() as i32;
            let u1 = u0 + 1; let v1 = v0 + 1;
            if u0 < 0 || v0 < 0 || u1 >= iw as i32 || v1 >= ih as i32 { continue; }
            let dx = u - u0 as f32; let dy = v - v0 as f32;
            let p00 = img.get_pixel(u0 as u32, v0 as u32).0[0] as f32;
            let p10 = img.get_pixel(u1 as u32, v0 as u32).0[0] as f32;
            let p01 = img.get_pixel(u0 as u32, v1 as u32).0[0] as f32;
            let p11 = img.get_pixel(u1 as u32, v1 as u32).0[0] as f32;
            let val = p00*(1.0-dx)*(1.0-dy) + p10*dx*(1.0-dy) + p01*(1.0-dx)*dy + p11*dx*dy;
            out.put_pixel(x, y, Luma([val.round() as u8]));
        }
    }
    out
}

fn draw_thick_rect(img: &mut RgbImage, color: Rgb<u8>, x: u32, y: u32, w: u32, h: u32, thickness: u32) {
    for t in 0..thickness {
        if w > 2 * t && h > 2 * t {
            draw_hollow_rect_mut(img, Rect::at((x + t) as i32, (y + t) as i32).of_size(w - 2 * t, h - 2 * t), color);
        }
    }
}

fn draw_thick_cross(img: &mut RgbImage, color: Rgb<u8>, cx: i32, cy: i32, radius: i32, thickness: i32) {
    for d in -thickness..=thickness {
        for i in -radius..=radius {
            let px = cx + i; let py = cy + d;
            if px >= 0 && py >= 0 && px < img.width() as i32 && py < img.height() as i32 {
                img.put_pixel(px as u32, py as u32, color);
            }
            let px2 = cx + d; let py2 = cy + i;
            if px2 >= 0 && py2 >= 0 && px2 < img.width() as i32 && py2 < img.height() as i32 {
                img.put_pixel(px2 as u32, py2 as u32, color);
            }
        }
    }
}

fn dilate_mask(mask: &GrayImage, rx: i32, ry: i32) -> GrayImage {
    let (w, h) = (mask.width(), mask.height());
    let mut tmp = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut m = 0u8;
            for dx in -rx..=rx {
                let xx = x as i32 + dx;
                if xx < 0 || xx >= w as i32 { continue; }
                let v = mask.get_pixel(xx as u32, y).0[0];
                if v > m { m = v; }
            }
            tmp.put_pixel(x, y, Luma([m]));
        }
    }
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut m = 0u8;
            for dy in -ry..=ry {
                let yy = y as i32 + dy;
                if yy < 0 || yy >= h as i32 { continue; }
                let v = tmp.get_pixel(x, yy as u32).0[0];
                if v > m { m = v; }
            }
            out.put_pixel(x, y, Luma([m]));
        }
    }
    out
}

fn local_text_mask(gray: &GrayImage, half: i32, c: i64) -> GrayImage {
    let (w, h) = (gray.width(), gray.height());
    let wp = (w + 1) as usize;
    let mut integ = vec![0u64; wp * ((h + 1) as usize)];
    for y in 0..h as usize {
        let mut row_sum = 0u64;
        for x in 0..w as usize {
            row_sum += gray.get_pixel(x as u32, y as u32).0[0] as u64;
            integ[(y + 1) * wp + (x + 1)] = integ[y * wp + (x + 1)] + row_sum;
        }
    }
    let mut out = GrayImage::new(w, h);
    for y in 0..h as i32 {
        let y0 = (y - half).max(0) as usize;
        let y1 = (y + half + 1).min(h as i32) as usize;
        for x in 0..w as i32 {
            let x0 = (x - half).max(0) as usize;
            let x1 = (x + half + 1).min(w as i32) as usize;
            let area = ((x1 - x0) * (y1 - y0)) as i64;
            let sum = (integ[y1 * wp + x1] + integ[y0 * wp + x0]) as i64
                - (integ[y0 * wp + x1] + integ[y1 * wp + x0]) as i64;
            let mean = sum / area;
            let v = gray.get_pixel(x as u32, y as u32).0[0] as i64;
            out.put_pixel(x as u32, y as u32, Luma([if v < mean - c { 255 } else { 0 }]));
        }
    }
    out
}

fn draw_text_blocks(vis: &mut RgbImage, gray: &GrayImage, thickness: u32, w: u32, h: u32, out_dir: &PathBuf) -> Vec<(u32, u32, u32, u32)> {
    let smooth = median_filter(gray, 1, 1);
    let text = local_text_mask(&smooth, 16, 20);

    #[derive(Clone)]
    struct BB { min_x: u32, min_y: u32, max_x: u32, max_y: u32, count: u64 }

    // ---- pass 1: remove separator lines, cut lines, edge lines, speckles ----
    let labels1 = connected_components(&text, Connectivity::Four, Luma([0u8]));
    let mut b1: Vec<BB> = Vec::new();
    for (x, y, p) in labels1.enumerate_pixels() {
        let l = p.0[0] as usize;
        if l == 0 { continue; }
        if l >= b1.len() { b1.resize(l + 1, BB { min_x: w, min_y: h, max_x: 0, max_y: 0, count: 0 }); }
        let bb = &mut b1[l];
        bb.count += 1;
        if x < bb.min_x { bb.min_x = x; }
        if x > bb.max_x { bb.max_x = x; }
        if y < bb.min_y { bb.min_y = y; }
        if y > bb.max_y { bb.max_y = y; }
    }

    let mut cleaned = text.clone();
    for (x, y, p) in labels1.enumerate_pixels() {
        let l = p.0[0] as usize;
        if l == 0 { continue; }
        let bb = &b1[l];
        let bw = bb.max_x - bb.min_x + 1;
        let bh = bb.max_y - bb.min_y + 1;
        let thin_h = (bw as f64) > 0.9 * (w as f64) && (bh as f64) < 0.06 * (h as f64);
        let thin_v = (bh as f64) > 0.9 * (h as f64) && (bw as f64) < 0.06 * (w as f64);
        if thin_h || thin_v || bb.count < 30 {
            cleaned.put_pixel(x, y, Luma([0]));
        }
    }

    for y in 0..h {
        for x in 0..w {
            if x < 6 || y < 4 || x >= w - 6 || y >= h - 4 {
                cleaned.put_pixel(x, y, Luma([0]));
            }
        }
    }

    // ---- pass 2: dilate into blocks and draw ----
    let dilated = dilate_mask(&cleaned, 12, 8);
    let mask_path = out_dir.join("step5_blocks_mask.png");
    dilated.save(&mask_path).expect("save blocks mask");

    let labels2 = connected_components(&dilated, Connectivity::Four, Luma([0u8]));
    let mut b2: Vec<BB> = Vec::new();
    for (x, y, p) in labels2.enumerate_pixels() {
        let l = p.0[0] as usize;
        if l == 0 { continue; }
        if l >= b2.len() { b2.resize(l + 1, BB { min_x: w, min_y: h, max_x: 0, max_y: 0, count: 0 }); }
        let bb = &mut b2[l];
        bb.count += 1;
        if x < bb.min_x { bb.min_x = x; }
        if x > bb.max_x { bb.max_x = x; }
        if y < bb.min_y { bb.min_y = y; }
        if y > bb.max_y { bb.max_y = y; }
    }

    let min_area = (w as u64 * h as u64) / 100;
    let mut drawn = 0;
    let mut boxes: Vec<(u32, u32, u32, u32)> = Vec::new();
    for bb in b2.iter() {
        let bw = bb.max_x - bb.min_x + 1;
        let bh = bb.max_y - bb.min_y + 1;
        let frame = bw as f64 > 0.95 * w as f64 && bh as f64 > 0.9 * h as f64;
        if bb.count >= min_area && !frame {
            drawn += 1;
            println!("text_block       : {},{} {}x{}", bb.min_x, bb.min_y, bw, bh);
            draw_thick_rect(vis, Rgb([0u8, 0, 255]), bb.min_x, bb.min_y, bw, bh, thickness);
            boxes.push((bb.min_x, bb.min_y, bw, bh));
        }
    }
    println!("text_blocks_drawn: {drawn}");
    boxes
}

fn percentile(hist: &[u64; 256], n: u64, q: f64) -> u8 {
    let target = (q * n as f64) as u64;
    let mut acc = 0u64;
    for (i, cc) in hist.iter().enumerate() {
        acc += cc;
        if acc >= target { return i as u8; }
    }
    255
}

fn stretch_gray(g: &GrayImage, lo_q: f64, hi_q: f64) -> GrayImage {
    let mut hist = [0u64; 256];
    let mut n = 0u64;
    for p in g.pixels() { hist[p.0[0] as usize] += 1; n += 1; }
    let lo = percentile(&hist, n, lo_q) as i32;
    let hi = (percentile(&hist, n, hi_q) as i32).max(lo + 1);
    let mut out = GrayImage::new(g.width(), g.height());
    for (x, y, p) in g.enumerate_pixels() {
        let t = (p.0[0] as i32 - lo).max(0) as f64 * 255.0 / ((hi - lo) as f64);
        out.put_pixel(x, y, Luma([t.min(255.0) as u8]));
    }
    out
}

fn stretch_rgb(img: &RgbImage) -> RgbImage {
    let mut hist = [[0u64; 256]; 3];
    let mut n = 0u64;
    for p in img.pixels() { for c in 0..3 { hist[c][p.0[c] as usize] += 1; } n += 1; }
    let mut lo = [0i32; 3];
    let mut hi = [0i32; 3];
    for c in 0..3 {
        lo[c] = percentile(&hist[c], n, 0.01) as i32;
        hi[c] = (percentile(&hist[c], n, 0.99) as i32).max(lo[c] + 1);
    }
    let mut out = RgbImage::new(img.width(), img.height());
    for (x, y, p) in img.enumerate_pixels() {
        let mut q = [0u8; 3];
        for c in 0..3 {
            let t = (p.0[c] as i32 - lo[c]).max(0) as f64 * 255.0 / ((hi[c] - lo[c]) as f64);
            q[c] = t.min(255.0) as u8;
        }
        out.put_pixel(x, y, Rgb(q));
    }
    out
}

fn sauvola(g: &GrayImage, half: i32, k: f64) -> GrayImage {
    let (w, h) = (g.width(), g.height());
    let wp = (w + 1) as usize;
    let mut sum = vec![0u64; wp * ((h + 1) as usize)];
    let mut sumsq = vec![0u64; wp * ((h + 1) as usize)];
    for y in 0..h as usize {
        let mut rs = 0u64;
        let mut rsq = 0u64;
        for x in 0..w as usize {
            let v = g.get_pixel(x as u32, y as u32).0[0] as u64;
            rs += v;
            rsq += v * v;
            sum[(y + 1) * wp + (x + 1)] = sum[y * wp + (x + 1)] + rs;
            sumsq[(y + 1) * wp + (x + 1)] = sumsq[y * wp + (x + 1)] + rsq;
        }
    }
    let mut out = GrayImage::new(w, h);
    for y in 0..h as i32 {
        let y0 = (y - half).max(0) as usize;
        let y1 = (y + half + 1).min(h as i32) as usize;
        for x in 0..w as i32 {
            let x0 = (x - half).max(0) as usize;
            let x1 = (x + half + 1).min(w as i32) as usize;
            let area = ((x1 - x0) * (y1 - y0)) as f64;
            let s = (sum[y1 * wp + x1] + sum[y0 * wp + x0]) as f64 - (sum[y0 * wp + x1] + sum[y1 * wp + x0]) as f64;
            let sq = (sumsq[y1 * wp + x1] + sumsq[y0 * wp + x0]) as f64 - (sumsq[y0 * wp + x1] + sumsq[y1 * wp + x0]) as f64;
            let mean = s / area;
            let std = ((sq / area) - mean * mean).max(0.0).sqrt();
            let t = mean * (1.0 + k * (std / 128.0 - 1.0));
            let v = g.get_pixel(x as u32, y as u32).0[0] as f64;
            out.put_pixel(x as u32, y as u32, Luma([if v > t { 255 } else { 0 }]));
        }
    }
    out
}

fn warp_perspective_rgb(img: &RgbImage, hm: &[f32; 9], out_w: u32, out_h: u32) -> RgbImage {
    let mut out = RgbImage::from_pixel(out_w, out_h, Rgb([255u8, 255, 255]));
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    for y in 0..out_h {
        for x in 0..out_w {
            let xf = x as f32; let yf = y as f32;
            let wden = hm[6]*xf + hm[7]*yf + hm[8];
            if wden.abs() < 1e-6 { continue; }
            let u = (hm[0]*xf + hm[1]*yf + hm[2]) / wden;
            let v = (hm[3]*xf + hm[4]*yf + hm[5]) / wden;
            let u0 = u.floor() as i32; let v0 = v.floor() as i32;
            let u1 = u0 + 1; let v1 = v0 + 1;
            if u0 < 0 || v0 < 0 || u1 >= iw as i32 || v1 >= ih as i32 { continue; }
            let dx = u - u0 as f32; let dy = v - v0 as f32;
            let mut q = [0u8; 3];
            for c in 0..3 {
                let p00 = img.get_pixel(u0 as u32, v0 as u32).0[c] as f32;
                let p10 = img.get_pixel(u1 as u32, v0 as u32).0[c] as f32;
                let p01 = img.get_pixel(u0 as u32, v1 as u32).0[c] as f32;
                let p11 = img.get_pixel(u1 as u32, v1 as u32).0[c] as f32;
                let val = p00*(1.0-dx)*(1.0-dy) + p10*dx*(1.0-dy) + p01*(1.0-dx)*dy + p11*dx*dy;
                q[c] = val.round() as u8;
            }
            out.put_pixel(x, y, Rgb(q));
        }
    }
    out
}
fn find_tesseract() -> Option<String> {
    let candidates = vec![
        "tesseract".to_string(),
        "C:\\Program Files\\Tesseract-OCR\\tesseract.exe".to_string(),
        "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe".to_string(),
    ];
    for c in candidates {
        if let Ok(o) = std::process::Command::new(&c).arg("--version").output() {
            if o.status.success() {
                return Some(c);
            }
        }
    }
    None
}

fn run_ocr(tess: &str, img: &Path, tsv: bool) -> String {
    let mut cmd = std::process::Command::new(tess);
    cmd.arg(img).arg("stdout");
    if tsv {
        cmd.arg("tsv");
    }
    let out = cmd.output().expect("run tesseract");
    String::from_utf8_lossy(&out.stdout).to_string()
}

fn ocr_stats(tsv: &str) -> (f64, usize) {
    let mut sum = 0f64;
    let mut n = 0usize;
    for line in tsv.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() >= 12 {
            if let Ok(c) = f[10].parse::<f64>() {
                if c >= 0.0 {
                    sum += c;
                    n += 1;
                }
            }
        }
    }
    if n > 0 { (sum / n as f64, n) } else { (0.0, 0) }
}
fn pdf_escape_bytes(s: &str) -> Vec<u8> {
    let mut out = Vec::new();
    for ch in s.chars() {
        let b = match ch {
            '\u{2018}' => 0x91u8,
            '\u{2019}' => 0x92u8,
            '\u{201C}' => 0x93u8,
            '\u{201D}' => 0x94u8,
            '\u{2013}' => 0x96u8,
            '\u{2014}' => 0x97u8,
            '\u{2026}' => 0x85u8,
            c if (c as u32) >= 32 && (c as u32) <= 126 => c as u8,
            _ => b'?',
        };
        match b {
            b'(' | b')' | b'\\' => {
                out.push(b'\\');
                out.push(b);
            }
            _ => out.push(b),
        }
    }
    out
}

fn build_searchable_pdf(w: u32, h: u32, jpeg: &[u8], words: &[(i32, i32, i32, i32, String)]) -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(format!("q\n{w} 0 0 {h} 0 0 cm\n/Im0 Do\nQ\n").as_bytes());
    emit_text_lines(&mut content, h as i32, words, w as f64);

    let mut out = Vec::new();
    out.extend_from_slice(b"%PDF-1.4\n");
    let mut offsets: Vec<usize> = Vec::new();

    offsets.push(out.len());
    out.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    offsets.push(out.len());
    out.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    offsets.push(out.len());
    out.extend_from_slice(
        format!("3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] /Resources << /XObject << /Im0 5 0 R >> /Font << /F1 6 0 R >> >> /Contents 4 0 R >>\nendobj\n").as_bytes(),
    );

    offsets.push(out.len());
    out.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    out.extend_from_slice(&content);
    out.extend_from_slice(b"\nendstream\nendobj\n");

    offsets.push(out.len());
    out.extend_from_slice(
        format!("5 0 obj\n<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {} >>\nstream\n", jpeg.len()).as_bytes(),
    );
    out.extend_from_slice(jpeg);
    out.extend_from_slice(b"\nendstream\nendobj\n");

    offsets.push(out.len());
    out.extend_from_slice(b"6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n");

    let xref_pos = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n", offsets.len() + 1).as_bytes());
    out.extend_from_slice(b"0000000000 65535 f \n");
    for off in &offsets {
        out.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    out.extend_from_slice(
        format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", offsets.len() + 1, xref_pos).as_bytes(),
    );

    out
}
fn find_separator_rows(gray: &GrayImage) -> Vec<u32> {
    let (w, h) = (gray.width(), gray.height());
    if h < 300 { return Vec::new(); }
    let mut centers = Vec::new();
    let y_start = h / 5;
    let y_end = h - h / 5;
    let mut y = y_start;
    while y < y_end {
        let mut dark = 0u32;
        for x in 0..w {
            if gray.get_pixel(x, y).0[0] < 225 {
                dark += 1;
            }
        }
        if (dark as f64) > 0.85 * (w as f64) {
            let start = y;
            while y < h {
                let mut d2 = 0u32;
                for x in 0..w {
                    if gray.get_pixel(x, y).0[0] < 225 {
                        d2 += 1;
                    }
                }
                if (d2 as f64) <= 0.85 * (w as f64) {
                    break;
                }
                y += 1;
            }
            let height = y - start;
            if (2..80).contains(&height) {
                centers.push((start + y) / 2);
            }
        } else {
            y += 1;
        }
    }
    centers
}
fn build_multi_pdf(pages: &[(u32, u32, Vec<u8>, Vec<(i32, i32, i32, i32, String)>)]) -> Vec<u8> {
    let n = pages.len();
    let mut contents = Vec::new();
    for (w, h, _, words) in pages {
        let mut content = Vec::new();
        content.extend_from_slice(format!("q\n{w} 0 0 {h} 0 0 cm\n/Im0 Do\nQ\n").as_bytes());
        emit_text_lines(&mut content, *h as i32, words, *w as f64);
        contents.push(content);
    }

    let mut out = Vec::new();
    out.extend_from_slice(b"%PDF-1.4\n");
    let mut offsets: Vec<usize> = Vec::new();

    offsets.push(out.len());
    out.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let kids = (0..n).map(|i| format!("{} 0 R", 4 + 3 * i)).collect::<Vec<_>>().join(" ");
    offsets.push(out.len());
    out.extend_from_slice(format!("2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {n} >>\nendobj\n").as_bytes());

    offsets.push(out.len());
    out.extend_from_slice(b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n");

    for i in 0..n {
        let (w, h, jpeg, _) = &pages[i];
        let page_obj = 4 + 3 * i;
        let content_obj = page_obj + 1;
        let img_obj = page_obj + 2;

        offsets.push(out.len());
        out.extend_from_slice(format!("{page_obj} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] /Resources << /XObject << /Im0 {img_obj} 0 R >> /Font << /F1 3 0 R >> >> /Contents {content_obj} 0 R >>\nendobj\n").as_bytes());

        offsets.push(out.len());
        out.extend_from_slice(format!("{content_obj} 0 obj\n<< /Length {} >>\nstream\n", contents[i].len()).as_bytes());
        out.extend_from_slice(&contents[i]);
        out.extend_from_slice(b"\nendstream\nendobj\n");

        offsets.push(out.len());
        out.extend_from_slice(format!("{img_obj} 0 obj\n<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {} >>\nstream\n", jpeg.len()).as_bytes());
        out.extend_from_slice(jpeg);
        out.extend_from_slice(b"\nendstream\nendobj\n");
    }

    let xref_pos = out.len();
    let total = offsets.len() + 1;
    out.extend_from_slice(format!("xref\n0 {total}\n").as_bytes());
    out.extend_from_slice(b"0000000000 65535 f \n");
    for off in &offsets {
        out.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    out.extend_from_slice(format!("trailer\n<< /Size {total} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n").as_bytes());
    out
}
fn helv_width(b: u8) -> f64 {
    const W: [u16; 95] = [
        278,278,355,556,556,889,667,191,333,333,389,584,278,333,278,278,
        556,556,556,556,556,556,556,556,556,556,278,278,584,584,584,556,
        1015,667,667,722,722,667,611,778,722,278,500,667,556,833,722,778,
        667,778,722,667,611,722,667,944,667,667,611,278,278,278,469,556,
        333,556,556,500,556,556,278,556,556,222,222,500,222,833,556,556,
        556,556,333,500,278,556,500,722,500,500,500,334,260,334,584];
    if (32..=126).contains(&b) { W[(b - 32) as usize] as f64 } else { 556.0 }
}

fn word_em(text: &str) -> f64 {
    let mut sum = 0.0;
    for b in pdf_escape_bytes(text) {
        sum += helv_width(b);
    }
    sum / 1000.0
}
fn emit_text_lines(content: &mut Vec<u8>, h: i32, words: &[(i32, i32, i32, i32, String)], page_w: f64) {
    let filtered: Vec<&(i32, i32, i32, i32, String)> = words
        .iter()
        .filter(|w| (w.2 as f64) < 0.9 * page_w)
        .collect();

    let mut i = 0;
    while i < filtered.len() {
        let y_line = filtered[i].1;
        let mut j = i;
        while j < filtered.len() && filtered[j].1 == y_line {
            j += 1;
        }
        let line = &filtered[i..j];
        let bh_line = line.iter().map(|w| w.3).max().unwrap() as f64;
        let x0 = line[0].0 as f64;
        let right = line.iter().map(|w| w.0 + w.2).max().unwrap() as f64;
        let joined = line.iter().map(|w| w.4.as_str()).collect::<Vec<&str>>().join(" ");
        let size = (bh_line * 0.8).clamp(4.0, 40.0);
        let natural = (word_em(&joined) * size).max(1.0);
        let tz = ((right - x0) / natural * 100.0).clamp(10.0, 400.0);
        let yb = (h - y_line) as f64 - bh_line * 0.8;
        content.extend_from_slice(
            format!("BT\n3 Tr\n/F1 {size:.2} Tf\n{tz:.1} Tz\n1 0 0 1 {x0:.1} {yb:.1} Tm\n(").as_bytes(),
        );
        content.extend_from_slice(&pdf_escape_bytes(&joined));
        content.extend_from_slice(b") Tj\nET\n");
        i = j;
    }
}
struct Tps {
    r: f64,
    px: Vec<f64>, py: Vec<f64>,
    wx: Vec<f64>, wy: Vec<f64>,
    a0x: f64, axx: f64, ayx: f64,
    a0y: f64, axy: f64, ayy: f64,
}

fn kern(d: f64, r: f64) -> f64 {
    if r > 0.0 {
        let t = d / r;
        if t < 1.0 {
            let u = 1.0 - t;
            u * u * u * u * (1.0 + 4.0 * t)
        } else {
            0.0
        }
    } else {
        let r2 = d * d;
        if r2 > 1e-12 { r2 * r2.ln() } else { 0.0 }
    }
}


fn tps_fit_r(ctrl: &[(f64, f64)], val: &[(f64, f64)], radius: f64) -> Tps {
    let n = ctrl.len();
    let m = n + 3;
    let mut k = vec![vec![0.0f64; m]; m];
    let mut bx = vec![0.0f64; m];
    let mut by = vec![0.0f64; m];
    for i in 0..n {
        for j in 0..n {
            let dx = ctrl[i].0 - ctrl[j].0;
            let dy = ctrl[i].1 - ctrl[j].1;
            let d = (dx * dx + dy * dy).sqrt();
            k[i][j] = kern(d, radius);
        }
        k[i][n] = 1.0;
        k[i][n + 1] = ctrl[i].0;
        k[i][n + 2] = ctrl[i].1;
        k[n][i] = 1.0;
        k[n + 1][i] = ctrl[i].0;
        k[n + 2][i] = ctrl[i].1;
        bx[i] = val[i].0;
        by[i] = val[i].1;
    }
    for i in 0..m {
        let mut piv = i;
        for r in i + 1..m {
            if k[r][i].abs() > k[piv][i].abs() { piv = r; }
        }
        k.swap(i, piv); bx.swap(i, piv); by.swap(i, piv);
        if k[i][i].abs() < 1e-9 { continue; }
        for r in 0..m {
            if r == i { continue; }
            let f = k[r][i] / k[i][i];
            for c2 in i..m { k[r][c2] -= f * k[i][c2]; }
            bx[r] -= f * bx[i];
            by[r] -= f * by[i];
        }
    }
    let mut s = vec![0.0f64; m];
    let mut t = vec![0.0f64; m];
    for i in 0..m {
        s[i] = bx[i] / k[i][i];
        t[i] = by[i] / k[i][i];
    }
    Tps {
        px: ctrl.iter().map(|p| p.0).collect(),
        py: ctrl.iter().map(|p| p.1).collect(),
        wx: s[..n].to_vec(), wy: t[..n].to_vec(),
        r: radius,
        a0x: s[n], axx: s[n + 1], ayx: s[n + 2],
        a0y: t[n], axy: t[n + 1], ayy: t[n + 2],
    }
}

fn tps_eval(tps: &Tps, x: f64, y: f64) -> (f64, f64) {
    let mut ox = tps.a0x + tps.axx * x + tps.ayx * y;
    let mut oy = tps.a0y + tps.axy * x + tps.ayy * y;
    for i in 0..tps.px.len() {
        let dx = x - tps.px[i];
        let dy = y - tps.py[i];
        {
            let uu = kern((dx * dx + dy * dy).sqrt(), tps.r);
            ox += tps.wx[i] * uu;
            oy += tps.wy[i] * uu;
        }
    }
    (ox, oy)
}

fn sample_gray(img: &GrayImage, u: f64, v: f64) -> f64 {
    let (iw, ih) = (img.width() as i32, img.height() as i32);
    let u0 = u.floor() as i32;
    let v0 = v.floor() as i32;
    if u0 < 0 || v0 < 0 || u0 + 1 >= iw || v0 + 1 >= ih { return 255.0; }
    let dx = u - u0 as f64;
    let dy = v - v0 as f64;
    let p00 = img.get_pixel(u0 as u32, v0 as u32).0[0] as f64;
    let p10 = img.get_pixel(u0 as u32 + 1, v0 as u32).0[0] as f64;
    let p01 = img.get_pixel(u0 as u32, v0 as u32 + 1).0[0] as f64;
    let p11 = img.get_pixel(u0 as u32 + 1, v0 as u32 + 1).0[0] as f64;
    p00 * (1.0 - dx) * (1.0 - dy) + p10 * dx * (1.0 - dy) + p01 * (1.0 - dx) * dy + p11 * dx * dy
}

fn selftest13(out_dir: &PathBuf) {
    let (w, h) = (800u32, 1000u32);
    let mut flat = GrayImage::from_pixel(w, h, Luma([255u8]));
    for y in (100..900).step_by(50) {
        for t in 0..3i32 {
            for x in 60..740 { flat.put_pixel(x, (y as i32 + t - 1) as u32, Luma([0])); }
        }
    }
    for x in (100..750).step_by(100) {
        for t in 0..3i32 {
            for y in 60..940 { flat.put_pixel((x as i32 + t - 1) as u32, y, Luma([0])); }
        }
    }

    let pi = std::f64::consts::PI;
    let distort = |x: f64, y: f64| {
        let u = x / w as f64;
        let v = y / h as f64;
        (
            x + 18.0 * (pi * v).sin() * (2.0 * pi * u).sin(),
            y + 12.0 * (2.0 * pi * u).sin() * (pi * v).sin(),
        )
    };

    let mut curved = GrayImage::from_pixel(w, h, Luma([255u8]));
    for y in 0..h {
        for x in 0..w {
            let qx = x as f64;
            let qy = y as f64;
            let mut px = qx;
            let mut py = qy;
            for _ in 0..4 {
                let (dx, dy) = distort(px, py);
                px += qx - dx;
                py += qy - dy;
            }
            curved.put_pixel(x, y, Luma([sample_gray(&flat, px, py).round() as u8]));
        }
    }

    let mut ctrl: Vec<(f64, f64)> = Vec::new();
    let mut val: Vec<(f64, f64)> = Vec::new();
    for yy in [40.0, 180.0, 320.0, 460.0, 600.0, 740.0, 880.0, 960.0] {
        for xx in [40.0, 130.0, 220.0, 310.0, 400.0, 490.0, 580.0, 670.0, 760.0] {
            let (dx, dy) = distort(xx, yy);
            ctrl.push((xx, yy));
            val.push((dx, dy));
        }
    }
    let tps = tps_fit_r(&ctrl, &val, 0.45 * (w.min(h)) as f64);

    let mut dewarped = GrayImage::from_pixel(w, h, Luma([255u8]));
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = tps_eval(&tps, x as f64, y as f64);
            dewarped.put_pixel(x, y, Luma([sample_gray(&curved, sx, sy).round() as u8]));
        }
    }

    let mut sum = 0.0f64;
    let mut n = 0u64;
    let mut bad = 0u64;
    for y in 80..920 {
        for x in 60..740 {
            let d = (dewarped.get_pixel(x, y).0[0] as i32 - flat.get_pixel(x, y).0[0] as i32).abs();
            sum += d as f64;
            n += 1;
            if d > 64 { bad += 1; }
        }
    }
    let mean = sum / n as f64;
    let badf = bad as f64 / n as f64 * 100.0;

    // mesh overlay: destination grid mapped through the model onto the curved page
    let mut mesh = RgbImage::from_fn(w, h, |x, y| {
        let v = curved.get_pixel(x, y).0[0];
        Rgb([v, v, v])
    });
    for gy in (100..=900).step_by(50) {
        let yy = gy as f64;
        let mut prev: Option<(i32, i32)> = None;
        for gx in 0..=80 {
            let xx = 40.0 + (720.0 * gx as f64) / 80.0;
            let (sx, sy) = tps_eval(&tps, xx, yy);
            let pt = (sx.round() as i32, sy.round() as i32);
            if let Some(p) = prev {
                let steps = ((pt.0 - p.0).abs().max((pt.1 - p.1).abs())).max(1);
                for s in 0..=steps {
                    let qx = p.0 + (pt.0 - p.0) * s / steps;
                    let qy = p.1 + (pt.1 - p.1) * s / steps;
                    if qx >= 0 && qy >= 0 && qx < w as i32 && qy < h as i32 {
                        mesh.put_pixel(qx as u32, qy as u32, Rgb([0u8, 255, 255]));
                    }
                }
            }
            prev = Some(pt);
        }
    }
    for gx in (100..=700).step_by(100) {
        let xx = gx as f64;
        let mut prev: Option<(i32, i32)> = None;
        for gy in 0..=100 {
            let yy = 40.0 + (920.0 * gy as f64) / 100.0;
            let (sx, sy) = tps_eval(&tps, xx, yy);
            let pt = (sx.round() as i32, sy.round() as i32);
            if let Some(p) = prev {
                let steps = ((pt.0 - p.0).abs().max((pt.1 - p.1).abs())).max(1);
                for s in 0..=steps {
                    let qx = p.0 + (pt.0 - p.0) * s / steps;
                    let qy = p.1 + (pt.1 - p.1) * s / steps;
                    if qx >= 0 && qy >= 0 && qx < w as i32 && qy < h as i32 {
                        mesh.put_pixel(qx as u32, qy as u32, Rgb([0u8, 255, 255]));
                    }
                }
            }
            prev = Some(pt);
        }
    }

    curved.save(out_dir.join("step13_curved.png")).expect("save curved");
    dewarped.save(out_dir.join("step13_dewarped.png")).expect("save dewarped");
    mesh.save(out_dir.join("step13_mesh.png")).expect("save mesh");

    let mut bow_dew = 0.0f64;
    let mut bow_cur = 0.0f64;
    for y0 in (100..900).step_by(50) {
        let mut mn_d = f64::INFINITY;
        let mut mx_d = f64::NEG_INFINITY;
        let mut mn_c = f64::INFINITY;
        let mut mx_c = f64::NEG_INFINITY;
        for x in (100..700).step_by(10) {
            let mut sw_d = 0.0;
            let mut sy_d = 0.0;
            let mut sw_c = 0.0;
            let mut sy_c = 0.0;
            for dy in -16i32..=16 {
                let yy = (y0 as i32 + dy) as u32;
                let vd = 255.0 - dewarped.get_pixel(x, yy).0[0] as f64;
                let vc = 255.0 - curved.get_pixel(x, yy).0[0] as f64;
                sw_d += vd;
                sy_d += vd * dy as f64;
                sw_c += vc;
                sy_c += vc * dy as f64;
            }
            if sw_d > 0.0 {
                let c = sy_d / sw_d;
                mn_d = mn_d.min(c);
                mx_d = mx_d.max(c);
            }
            if sw_c > 0.0 {
                let c = sy_c / sw_c;
                mn_c = mn_c.min(c);
                mx_c = mx_c.max(c);
            }
        }
        bow_dew = bow_dew.max(mx_d - mn_d);
        bow_cur = bow_cur.max(mx_c - mn_c);
    }

    println!("selftest13 mean_diff : {mean:.2} (informational, resampling floor)");
    println!("selftest13 bad_pct   : {badf:.2}");
    println!("selftest13 curved_bow: {bow_cur:.2}  <- how bent the input is");
    println!("selftest13 dewarp_bow: {bow_dew:.2}  <- how straight the output is");
    if bow_dew < 2.0 && badf < 5.0 {
        println!("STEP13_OK");
    } else {
        println!("STEP13_FAIL");
    }
}
fn smooth_line(pts: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let fit = |q: &[(f64, f64)]| -> Option<(f64, f64, f64)> {
        let n = q.len() as f64;
        if n < 5.0 {
            return None;
        }
        let (mut sx, mut sx2, mut sx3, mut sx4) = (0.0, 0.0, 0.0, 0.0);
        let (mut sy, mut sxy, mut sx2y) = (0.0, 0.0, 0.0);
        for (x, y) in q {
            let x2 = x * x;
            sx += x;
            sx2 += x2;
            sx3 += x2 * x;
            sx4 += x2 * x2;
            sy += y;
            sxy += x * y;
            sx2y += x2 * y;
        }
        let mut a = [[n, sx, sx2, sy], [sx, sx2, sx3, sxy], [sx2, sx3, sx4, sx2y]];
        for i in 0..3 {
            let mut p = i;
            for j in (i + 1)..3 {
                if a[j][i].abs() > a[p][i].abs() {
                    p = j;
                }
            }
            if a[p][i].abs() < 1e-9 {
                return None;
            }
            a.swap(i, p);
            for j in 0..3 {
                if j != i {
                    let f = a[j][i] / a[i][i];
                    for k in i..4 {
                        a[j][k] -= f * a[i][k];
                    }
                }
            }
        }
        Some((a[2][3] / a[2][2], a[1][3] / a[1][1], a[0][3] / a[0][0]))
    };
    let Some(coef) = fit(pts) else {
        return pts.to_vec();
    };
    let keep: Vec<(f64, f64)> = pts
        .iter()
        .copied()
        .filter(|(x, y)| (coef.0 * x * x + coef.1 * x + coef.2 - y).abs() < 4.0)
        .collect();
    let coef = fit(&keep).unwrap_or(coef);
    let x0 = pts.first().unwrap().0;
    let x1 = pts.last().unwrap().0;
    (0..=8)
        .map(|k| {
            let x = x0 + (x1 - x0) * k as f64 / 8.0;
            (x, coef.0 * x * x + coef.1 * x + coef.2)
        })
        .collect()
}

fn baselines_from_tsv(gray: &GrayImage) -> Vec<Vec<(f64, f64)>> {
    let tmp = std::env::temp_dir().join(format!("scanrs_tsvin_{}.png", std::process::id()));
    if gray.save(&tmp).is_err() {
        return Vec::new();
    }
    let outbase = std::env::temp_dir().join(format!("scanrs_tsvout_{}", std::process::id()));
    let st = std::process::Command::new("tesseract")
        .arg(tmp.to_string_lossy().to_string())
        .arg(outbase.to_string_lossy().to_string())
        .arg("tsv")
        .output();
    let _ = std::fs::remove_file(&tmp);
    let Ok(st) = st else {
        return Vec::new();
    };
    if !st.status.success() {
        return Vec::new();
    }
    let tsvpath = format!("{}.tsv", outbase.to_string_lossy());
    let tsv = std::fs::read_to_string(&tsvpath).unwrap_or_default();
    let _ = std::fs::remove_file(&tsvpath);
    let mut groups: std::collections::BTreeMap<(String, String, String, String), Vec<(f64, f64)>> =
        std::collections::BTreeMap::new();
    for ln in tsv.lines().skip(1) {
        let f: Vec<&str> = ln.split('\t').collect();
        if f.len() < 12 {
            continue;
        }
        let conf: f32 = f[10].parse().unwrap_or(-1.0);
        let text = f[11].trim();
        if conf < 0.0 || text.is_empty() {
            continue;
        }
        let left: i32 = f[6].parse().unwrap_or(0);
        let top: i32 = f[7].parse().unwrap_or(0);
        let wd: i32 = f[8].parse().unwrap_or(0);
        let ht: i32 = f[9].parse().unwrap_or(0);
        let key = (
            f[1].to_string(),
            f[2].to_string(),
            f[3].to_string(),
            f[4].to_string(),
        );
        groups
            .entry(key)
            .or_default()
            .push((left as f64 + wd as f64 / 2.0, (top + ht) as f64));
    }
    let mut lines: Vec<Vec<(f64, f64)>> = groups
        .into_values()
        .filter(|v| v.len() >= 3)
        .map(|mut v| {
            v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            smooth_line(&v)
        })
        .collect();
    lines.sort_by_key(|v| v[0].1 as i64);
    lines
}

fn extract_baselines(gray: &GrayImage) -> Vec<Vec<(f64, f64)>> {
    let smooth = median_filter(gray, 1, 1);
    let text = local_text_mask(&smooth, 16, 20);
    extract_baselines_mask(gray, &text, false)
}

fn extract_baselines_mask(gray: &GrayImage, text: &GrayImage, relaxed: bool) -> Vec<Vec<(f64, f64)>> {
    let (w, h) = (gray.width(), gray.height());
    let dil = dilate_mask(text, 10, 3);
    let labels = connected_components(&dil, Connectivity::Four, Luma([0u8]));

    #[derive(Clone)]
    struct LB { min_x: u32, max_x: u32, min_y: u32, max_y: u32, count: u64 }
    let mut lbs: Vec<LB> = Vec::new();
    for (x, y, p) in labels.enumerate_pixels() {
        let l = p.0[0] as usize;
        if l == 0 { continue; }
        if l >= lbs.len() {
            lbs.resize(l + 1, LB { min_x: w, max_x: 0, min_y: h, max_y: 0, count: 0 });
        }
        let lb = &mut lbs[l];
        lb.count += 1;
        if x < lb.min_x { lb.min_x = x; }
        if x > lb.max_x { lb.max_x = x; }
        if y < lb.min_y { lb.min_y = y; }
        if y > lb.max_y { lb.max_y = y; }
    }

    if relaxed {
        let mut top: Vec<(u64, u32, u32)> = lbs
            .iter()
            .map(|lb| {
                (
                    lb.count,
                    lb.max_x.saturating_sub(lb.min_x) + 1,
                    lb.max_y.saturating_sub(lb.min_y) + 1,
                )
            })
            .collect();
        top.sort_by(|a, b| b.0.cmp(&a.0));
        for t in top.iter().take(6) {
            println!("baselines_dbg        : count={} bw={} bh={}", t.0, t.1, t.2);
        }
    }
    let mut lines: Vec<Vec<(f64, f64)>> = Vec::new();
    for lb in lbs.iter() {
        let bw = lb.max_x.saturating_sub(lb.min_x) + 1;
        let bh = lb.max_y.saturating_sub(lb.min_y) + 1;
        let min_count = (w as u64 * h as u64) / if relaxed { 500 } else { 200 };
        let min_bw = (w as f64) / if relaxed { 4.0 } else { 3.0 };
        let max_bh = (h as f64) / if relaxed { 6.0 } else { 8.0 };
        if lb.count < min_count { continue; }
        if (bw as f64) < min_bw { continue; }
        if (bh as f64) > max_bh { continue; }

        let mut pts = Vec::new();
        for k in 0..7 {
            let xc = lb.min_x + (k * (lb.max_x - lb.min_x)) / 6;
            let mut sw = 0.0;
            let mut sy = 0.0;
            for x in xc.saturating_sub(3)..=(xc + 3).min(w - 1) {
                for y in lb.min_y..=lb.max_y {
                    if text.get_pixel(x, y).0[0] > 0 {
                        sw += 1.0;
                        sy += y as f64;
                    }
                }
            }
            if sw > 5.0 {
                pts.push((xc as f64, sy / sw));
            }
        }
        if pts.len() >= 5 {
            lines.push(pts);
        }
    }
    lines
}

fn dewarp_stage(gray: &GrayImage, rgb: &RgbImage, out_dir: &PathBuf) -> (GrayImage, RgbImage) {
    let (w, h) = (gray.width(), gray.height());
    let mut lines = extract_baselines(gray);
    if lines.len() < 3 {
        println!("baselines_fallback   : masks failed, using tesseract line geometry");
        lines = baselines_from_tsv(gray);
        println!("baselines_fallback   : tsv lines = {}", lines.len());
    }

    let bows: Vec<f64> = lines.iter().map(|line| line_bow(line)).collect();
    let mut bs = bows.clone();
    bs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let bow_before = if bs.is_empty() { 0.0 } else { bs[bs.len() / 2] };
    if bow_before < 3.0 {
        println!("dewarp_skipped   : straight (median bow {bow_before:.2})");
        return (gray.clone(), rgb.clone());
    }
    let bow_cap = bow_before * 3.0 + 2.0;

    let mut ctrl: Vec<(f64, f64)> = Vec::new();
    let mut val: Vec<(f64, f64)> = Vec::new();
    ctrl.push((0.0, 0.0)); val.push((0.0, 0.0));
    ctrl.push((w as f64, 0.0)); val.push((w as f64, 0.0));
    ctrl.push((w as f64, h as f64)); val.push((w as f64, h as f64));
    ctrl.push((0.0, h as f64)); val.push((0.0, h as f64));

    for (line, b) in lines.iter().zip(bows.iter()) {
        if *b > bow_cap {
            continue;
        }
        let ys: Vec<f64> = line.iter().map(|p| p.1).collect();
        let mut sorted = ys.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let my = sorted[sorted.len() / 2];
        for p in line {
            ctrl.push((p.0, my));
            val.push((p.0, p.1));
        }
    }

    // debug visual: baselines on the input
    let mut vis = RgbImage::from_fn(w, h, |x, y| {
        let v = gray.get_pixel(x, y).0[0];
        Rgb([v, v, v])
    });
    for line in &lines {
        for pair in line.windows(2) {
            let steps = ((pair[1].0 - pair[0].0).abs() as i32).max(1);
            for s in 0..=steps {
                let qx = (pair[0].0 + (pair[1].0 - pair[0].0) * s as f64 / steps as f64).round() as i32;
                let qy = (pair[0].1 + (pair[1].1 - pair[0].1) * s as f64 / steps as f64).round() as i32;
                for t in -1i32..=1 {
                    if qx >= 0 && qy + t >= 0 && qx < w as i32 && qy + t < h as i32 {
                        vis.put_pixel(qx as u32, (qy + t) as u32, Rgb([255u8, 0, 0]));
                    }
                }
            }
        }
    }
    vis.save(out_dir.join("step14_baselines.png")).expect("save baselines");

    let tps = tps_fit_r(&ctrl, &val, 0.45 * (w.min(h)) as f64);

    let mut g2 = GrayImage::from_pixel(w, h, Luma([255u8]));
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = tps_eval(&tps, x as f64, y as f64);
            g2.put_pixel(x, y, Luma([sample_gray(gray, sx, sy).round() as u8]));
        }
    }
    let mut r2 = RgbImage::from_pixel(w, h, Rgb([255u8, 255, 255]));
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = tps_eval(&tps, x as f64, y as f64);
            let u0 = sx.floor() as i32;
            let v0 = sy.floor() as i32;
            if u0 < 0 || v0 < 0 || u0 + 1 >= w as i32 || v0 + 1 >= h as i32 { continue; }
            let dx = sx - u0 as f64;
            let dy = sy - v0 as f64;
            let mut q = [0u8; 3];
            for cc in 0..3 {
                let p00 = rgb.get_pixel(u0 as u32, v0 as u32).0[cc] as f64;
                let p10 = rgb.get_pixel(u0 as u32 + 1, v0 as u32).0[cc] as f64;
                let p01 = rgb.get_pixel(u0 as u32, v0 as u32 + 1).0[cc] as f64;
                let p11 = rgb.get_pixel(u0 as u32 + 1, v0 as u32 + 1).0[cc] as f64;
                q[cc] = (p00 * (1.0 - dx) * (1.0 - dy) + p10 * dx * (1.0 - dy) + p01 * (1.0 - dx) * dy + p11 * dx * dy).round() as u8;
            }
            r2.put_pixel(x, y, Rgb(q));
        }
    }
    g2.save(out_dir.join("step14_dewarped.png")).expect("save dewarped");

    let mut bow_after = 0.0f64;
    for line in extract_baselines(&g2).iter() {
        let ys: Vec<f64> = line.iter().map(|p| p.1).collect();
        if ys.len() < 5 { continue; }
        let mn = ys.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        bow_after = bow_after.max(mx - mn);
    }

    println!("dewarp_lines      : {}", lines.len());
    println!("dewarp_bow_before : {bow_before:.2}");
    println!("dewarp_bow_after  : {bow_after:.2}");

    (g2, r2)
}
fn line_bow(line: &[(f64, f64)]) -> f64 {
    let n = line.len() as f64;
    if n < 3.0 {
        return 0.0;
    }
    let sx: f64 = line.iter().map(|p| p.0).sum();
    let sy: f64 = line.iter().map(|p| p.1).sum();
    let sxx: f64 = line.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = line.iter().map(|p| p.0 * p.1).sum();
    let den = n * sxx - sx * sx;
    if den.abs() < 1e-9 {
        return 0.0;
    }
    let a = (n * sxy - sx * sy) / den;
    let b = (sy - a * sx) / n;
    line.iter()
        .map(|p| (p.1 - (a * p.0 + b)).abs())
        .fold(0.0f64, f64::max)
}

fn auto_landmarks(gray: &GrayImage) -> (Vec<(f64, f64)>, Vec<(f64, f64)>, f64) {
    let (w, h) = (gray.width(), gray.height());
    let mut lines = extract_baselines(gray);
    if lines.len() < 3 {
        println!("baselines_fallback   : masks failed, using tesseract line geometry");
        lines = baselines_from_tsv(gray);
        println!("baselines_fallback   : tsv lines = {}", lines.len());
    }
    let mut ctrl: Vec<(f64, f64)> = Vec::new();
    let mut val: Vec<(f64, f64)> = Vec::new();
    ctrl.push((0.0, 0.0)); val.push((0.0, 0.0));
    ctrl.push((w as f64, 0.0)); val.push((w as f64, 0.0));
    ctrl.push((w as f64, h as f64)); val.push((w as f64, h as f64));
    ctrl.push((0.0, h as f64)); val.push((0.0, h as f64));
    let bows: Vec<f64> = lines.iter().map(|line| line_bow(line)).collect();
    let mut bs = bows.clone();
    bs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let bow = if bs.is_empty() { 0.0 } else { bs[bs.len() / 2] };
    let bow_cap = bow * 3.0 + 2.0;
    for (line, b) in lines.iter().zip(bows.iter()) {
        if *b > bow_cap {
            continue;
        }
        let ys: Vec<f64> = line.iter().map(|p| p.1).collect();
        let mut sorted = ys.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let my = sorted[sorted.len() / 2];
        for p in line {
            ctrl.push((p.0, my));
            val.push((p.0, p.1));
        }
    }
    (ctrl, val, bow)
}

fn tps_warp_mem(gray: &GrayImage, rgb: &RgbImage, ctrl: &[(f64, f64)], val: &[(f64, f64)]) -> (GrayImage, RgbImage) {
    let (w, h) = (gray.width(), gray.height());
    let tps = tps_fit_r(ctrl, val, 0.45 * (w.min(h)) as f64);
    let mut g2 = GrayImage::from_pixel(w, h, Luma([255u8]));
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = tps_eval(&tps, x as f64, y as f64);
            g2.put_pixel(x, y, Luma([sample_gray(gray, sx, sy).round() as u8]));
        }
    }
    let mut r2 = RgbImage::from_pixel(w, h, Rgb([255u8, 255, 255]));
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = tps_eval(&tps, x as f64, y as f64);
            let u0 = sx.floor() as i32;
            let v0 = sy.floor() as i32;
            if u0 < 0 || v0 < 0 || u0 + 1 >= w as i32 || v0 + 1 >= h as i32 { continue; }
            let dx = sx - u0 as f64;
            let dy = sy - v0 as f64;
            let mut q = [0u8; 3];
            for cc in 0..3 {
                let p00 = rgb.get_pixel(u0 as u32, v0 as u32).0[cc] as f64;
                let p10 = rgb.get_pixel(u0 as u32 + 1, v0 as u32).0[cc] as f64;
                let p01 = rgb.get_pixel(u0 as u32, v0 as u32 + 1).0[cc] as f64;
                let p11 = rgb.get_pixel(u0 as u32 + 1, v0 as u32 + 1).0[cc] as f64;
                q[cc] = (p00 * (1.0 - dx) * (1.0 - dy) + p10 * dx * (1.0 - dy) + p01 * (1.0 - dx) * dy + p11 * dx * dy).round() as u8;
            }
            r2.put_pixel(x, y, Rgb(q));
        }
    }
    (g2, r2)
}

fn tps_warp(gray: &GrayImage, rgb: &RgbImage, ctrl: &[(f64, f64)], val: &[(f64, f64)], out_dir: &PathBuf) -> (GrayImage, RgbImage) {
    let r = tps_warp_mem(gray, rgb, ctrl, val);
    r.0.save(out_dir.join("step15_dewarped.png")).expect("save gui dewarped");
    r
}

const GUI_HTML: &str = r#"<!doctype html>
<html><head><meta charset="utf-8"><title>scanrs landmark editor</title>
<style>body{background:#222;color:#eee;font-family:'Segoe UI',sans-serif}canvas{background:#fff;margin:8px}button{padding:8px 16px;font-size:14px}</style>
</head><body>
<h3>drag point: move | drag empty or shift+drag: pan | wheel: zoom | double-click: add | right-click: delete</h3>
<canvas id="cv" width="__W__" height="__H__"></canvas>
<div><button id="preview">Preview dewarp</button> <button id="apply">Apply and dewarp</button> <button id="reset">Reset points</button> <span id="status"></span></div><img id="pv" style="max-width:95%;margin:8px;border:1px solid #555">
<script>
let pts=[__POINTS__];let orig=JSON.parse(JSON.stringify(pts));
let drag=-1;
let scale=1,ox=0,oy=0;
let pan=null;
const cv=document.getElementById('cv');
const ctx=cv.getContext('2d');
const img=new Image();
img.onload=draw;
img.src='/img';
function draw(){
ctx.setTransform(1,0,0,1,0,0);
ctx.clearRect(0,0,cv.width,cv.height);
ctx.setTransform(scale,0,0,scale,ox,oy);
ctx.drawImage(img,0,0);
for(let i=0;i<pts.length;i++){
ctx.beginPath();ctx.arc(pts[i].sx,pts[i].sy,(i<4?8:6)/scale,0,7);
ctx.fillStyle=i<4?'rgba(0,200,0,0.9)':'rgba(255,0,255,0.9)';
ctx.fill();ctx.lineWidth=2/scale;ctx.strokeStyle='#000';ctx.stroke();
}
}
function pos(e){const r=cv.getBoundingClientRect();return[(e.clientX-r.left-ox)/scale,(e.clientY-r.top-oy)/scale];}
cv.oncontextmenu=e=>e.preventDefault();
cv.onwheel=e=>{e.preventDefault();const[x,y]=pos(e);const f=e.deltaY<0?1.2:1/1.2;scale*=f;const r=cv.getBoundingClientRect();ox=(e.clientX-r.left)-x*scale;oy=(e.clientY-r.top)-y*scale;draw();};
cv.onmousedown=e=>{const[x,y]=pos(e);
if(e.button===1||e.button===0&&e.shiftKey){pan=[e.clientX,e.clientY];return;}
if(e.button===2){const i=pts.findIndex(p=>Math.hypot(p.sx-x,p.sy-y)<9/scale);if(i>=4)pts.splice(i,1);draw();return;}
drag=pts.findIndex(p=>Math.hypot(p.sx-x,p.sy-y)<9/scale);
if(drag<0){pan=[e.clientX,e.clientY];}
};
cv.onmousemove=e=>{if(pan){ox+=e.clientX-pan[0];oy+=e.clientY-pan[1];pan=[e.clientX,e.clientY];draw();return;}
if(drag<0)return;const[x,y]=pos(e);if(drag>=4){pts[drag].sy=y;}else{pts[drag].sx=x;pts[drag].sy=y;}draw();};
window.onmouseup=()=>{drag=-1;pan=null;};
cv.ondblclick=e=>{const[x,y]=pos(e);pts.push({dx:x,dy:y,sx:x,sy:y});draw();};
document.getElementById('reset').onclick=()=>{pts=JSON.parse(JSON.stringify(orig));draw();document.getElementById('status').textContent='points reset';};
document.getElementById('preview').onclick=()=>{
fetch('/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(pts)})
.then(r=>r.blob()).then(b=>{document.getElementById('pv').src=URL.createObjectURL(b);document.getElementById('status').textContent='preview updated';});};
document.getElementById('apply').onclick=()=>{
fetch('/apply',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(pts)})
.then(()=>{document.getElementById('status').textContent='sent - returning to pipeline';});};
</script></body></html>"#;

fn gui_session(gray: &GrayImage, rgb: &RgbImage, ctrl: &[(f64, f64)], val: &[(f64, f64)]) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
    use image::ImageEncoder;
    use tiny_http::{Header, Response, Server};

    let mut png = Vec::new();
    {
        let cur = std::io::Cursor::new(&mut png);
        let enc = image::codecs::png::PngEncoder::new(cur);
        enc.write_image(rgb.as_raw(), rgb.width(), rgb.height(), image::ColorType::Rgb8.into()).expect("png encode");
    }

    let pts_json = ctrl
        .iter()
        .zip(val.iter())
        .map(|(d, s)| format!("{{\"dx\":{:.1},\"dy\":{:.1},\"sx\":{:.1},\"sy\":{:.1}}}", d.0, d.1, s.0, s.1))
        .collect::<Vec<_>>()
        .join(",");

    let html = GUI_HTML
        .replace("__POINTS__", &pts_json)
        .replace("__W__", &rgb.width().to_string())
        .replace("__H__", &rgb.height().to_string());

    let port = 8787u16;
    let server = Server::http(format!("127.0.0.1:{port}")).expect("gui server");
    let url = format!("http://127.0.0.1:{port}/");
    println!("GUI: open {url} - drag points, then Apply. Ctrl+C aborts.");
    let _ = std::process::Command::new("cmd").args(["/c", "start", "", &url]).spawn();

    let mut out_ctrl = ctrl.to_vec();
    let mut out_val = val.to_vec();
    for mut req in server.incoming_requests() {
        let u = req.url().to_string();
        if u == "/img" {
            let h = Header::from_bytes("Content-Type", "image/png").unwrap();
            let _ = req.respond(Response::from_data(png.clone()).with_header(h));
        } else if u == "/preview" {
            let mut body = String::new();
            let r = req.as_reader();
            let _ = r.read_to_string(&mut body);
            let mut pc: Vec<(f64, f64)> = Vec::new();
            let mut pv: Vec<(f64, f64)> = Vec::new();
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
                if let Some(arr) = v.as_array() {
                    for p in arr {
                        pc.push((p["dx"].as_f64().unwrap_or(0.0), p["dy"].as_f64().unwrap_or(0.0)));
                        pv.push((p["sx"].as_f64().unwrap_or(0.0), p["sy"].as_f64().unwrap_or(0.0)));
                    }
                }
            }
            let (_, prev) = tps_warp_mem(gray, rgb, &pc, &pv);
            let mut png2 = Vec::new();
            {
                let cur = std::io::Cursor::new(&mut png2);
                let enc = image::codecs::png::PngEncoder::new(cur);
                let _ = enc.write_image(prev.as_raw(), prev.width(), prev.height(), image::ColorType::Rgb8.into());
            }
            let h = Header::from_bytes("Content-Type", "image/png").unwrap();
            let _ = req.respond(Response::from_data(png2).with_header(h));
        } else if u == "/apply" {
            let mut body = String::new();
            let r = req.as_reader();
            let _ = r.read_to_string(&mut body);
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
                if let Some(arr) = v.as_array() {
                    out_ctrl.clear();
                    out_val.clear();
                    for p in arr {
                        out_ctrl.push((p["dx"].as_f64().unwrap_or(0.0), p["dy"].as_f64().unwrap_or(0.0)));
                        out_val.push((p["sx"].as_f64().unwrap_or(0.0), p["sy"].as_f64().unwrap_or(0.0)));
                    }
                }
            }
            let _ = req.respond(Response::from_string("ok"));
            break;
        } else {
            let h = Header::from_bytes("Content-Type", "text/html; charset=utf-8").unwrap();
            let _ = req.respond(Response::from_data(html.as_bytes().to_vec()).with_header(h));
        }
    }
    println!("gui_landmarks    : {}", out_ctrl.len());
    (out_ctrl, out_val)
}
fn decode_via_os(path: &std::path::Path) -> Option<image::RgbImage> {
    let tmp = std::env::temp_dir().join(format!("scanrs_osdecode_{}.png", std::process::id()));
    let src = path.to_string_lossy().replace('\'', "''");
    let tmps = tmp.to_string_lossy().replace('\'', "''");
    let cmd = format!(
        "Add-Type -AssemblyName System.Drawing; $i = [System.Drawing.Image]::FromFile('{src}'); $i.Save('{tmps}', [System.Drawing.Imaging.ImageFormat]::Png); $i.Dispose()"
    );
    let st = std::process::Command::new("powershell")
        .args(["-NoProfile", "-Command", &cmd])
        .output()
        .ok()?;
    if !st.status.success() {
        return None;
    }
    let im = image::open(&tmp).ok()?;
    let _ = std::fs::remove_file(&tmp);
    Some(im.to_rgb8())
}

fn load_input(path: &std::path::Path) -> image::RgbImage {
    match image::open(path) {
        Ok(im) => im.to_rgb8(),
        Err(e) => {
            println!("jpeg_fallback        : primary decoder failed ({e}), using zune-jpeg");
            let data = fs::read(path).expect("read input file");
            let mut dec = zune_jpeg::JpegDecoder::new(&data);
            let px = match dec.decode() {
                Ok(v) => v,
                Err(_) => {
                    println!("os_fallback          : zune failed too, using Windows codec (GDI+)");
                    return decode_via_os(path).expect("all decoders failed on this file");
                }
            };
            let (w, h) = dec.dimensions().expect("zune dims");
            let n = (w * h) as usize;
            let ch = if px.len() == n { 1usize } else if px.len() == n * 4 { 4 } else { 3 };
            let (w, h) = (w as u32, h as u32);
            let mut rgb = Vec::with_capacity((w * h * 3) as usize);
            for p in px.chunks(ch) {
                if ch == 1 {
                    rgb.push(p[0]); rgb.push(p[0]); rgb.push(p[0]);
                } else {
                    rgb.push(p[0]); rgb.push(p[1]); rgb.push(p[2]);
                }
            }
            image::RgbImage::from_raw(w, h, rgb).expect("rgb from raw")
        }
    }
}

fn cleanup_debug(out_dir: &PathBuf) {
    for n in [
        "step5_detect.png",
        "step6_warped.png",
        "step7_color_warped.png",
        "step7_magic_color.png",
        "step7_gray_stretch.png",
        "step7_equalized.png",
        "step7_bw_sauvola.png",
        "step7_bw_otsu.png",
        "step14_baselines.png",
        "step14_dewarped.png",
        "step15_dewarped.png",
        "text_lines.tsv",
    ] {
        let _ = fs::remove_file(out_dir.join(n));
    }
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let images_dir = PathBuf::from(std::env::var("LOCALAPPDATA").unwrap_or_else(|_| ".".to_string())).join("scanrs").join("images");

    let args: Vec<String> = env::args().collect();
    let mut input: Option<PathBuf> = None;
    let mut out_arg: Option<PathBuf> = None;
    let mut no_split = false;
    let mut selftest = false;
    let mut dewarp = true;
    let mut precurl = false;
    let mut gui = false;
    let mut open_out = false;
    let mut keep_debug = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                if i + 1 < args.len() {
                    out_arg = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--no-split" => { no_split = true; }
            "--selftest13" => { no_split = true; selftest = true; }
            "--dewarp" => { dewarp = true; }
            "--no-dewarp" => { dewarp = false; }
            "--gui" => { gui = true; }
            "--open" => { open_out = true; }
            "--debug" => { keep_debug = true; }
            "--precurl" => { precurl = true; }
            _ => {
                if input.is_none() {
                    input = Some(PathBuf::from(&args[i]));
                }
            }
        }
        i += 1;
    }
    let input = input.unwrap_or_else(|| images_dir.join("test_page.png"));
    let out_dir = match out_arg {
        Some(d) => d,
        None => {
            let stem = input
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "scan".to_string());
            let parent = input.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| root.clone());
            parent.join(format!("{stem}_scan"))
        }
    };
    fs::create_dir_all(&images_dir).expect("create images dir");
    fs::create_dir_all(&out_dir).expect("create out dir");

    if selftest {
        selftest13(&out_dir);
        return;
    }
    if !input.exists() { make_test_image(&input); }

    let img0 = load_input(&input);
    let img = if precurl {
        let rgb0 = img0.clone();
        let (iw, ih) = (rgb0.width(), rgb0.height());
        let pi = std::f64::consts::PI;
        let ax = (iw as f64 * 0.03).max(6.0);
        let ay = (ih as f64 * 0.02).max(4.0);
        let mut curled = RgbImage::from_pixel(iw, ih, Rgb([255u8, 255, 255]));
        for y in 0..ih {
            for x in 0..iw {
                let qx = x as f64;
                let qy = y as f64;
                let u = qx / iw as f64;
                let v = qy / ih as f64;
                let mut px = qx;
                let mut py = qy;
                for _ in 0..4 {
                    let dx = px + ax * (pi * v).sin() * (2.0 * pi * u).sin();
                    let dy = py + ay * (2.0 * pi * u).sin() * (pi * v).sin();
                    px += qx - dx;
                    py += qy - dy;
                }
                let u0 = px.floor() as i32;
                let v0 = py.floor() as i32;
                if u0 < 0 || v0 < 0 || u0 + 1 >= iw as i32 || v0 + 1 >= ih as i32 { continue; }
                let ddx = px - u0 as f64;
                let ddy = py - v0 as f64;
                let mut q = [0u8; 3];
                for cc in 0..3 {
                    let p00 = rgb0.get_pixel(u0 as u32, v0 as u32).0[cc] as f64;
                    let p10 = rgb0.get_pixel(u0 as u32 + 1, v0 as u32).0[cc] as f64;
                    let p01 = rgb0.get_pixel(u0 as u32, v0 as u32 + 1).0[cc] as f64;
                    let p11 = rgb0.get_pixel(u0 as u32 + 1, v0 as u32 + 1).0[cc] as f64;
                    q[cc] = (p00 * (1.0 - ddx) * (1.0 - ddy) + p10 * ddx * (1.0 - ddy) + p01 * (1.0 - ddx) * ddy + p11 * ddx * ddy).round() as u8;
                }
                curled.put_pixel(x, y, Rgb(q));
            }
        }
        image::DynamicImage::ImageRgb8(curled)
    } else {
        image::DynamicImage::ImageRgb8(img0)
    };
    let gray = img.to_luma8();

    // ---- step 11: multi-page split via recursive self-invocation ----
    if !no_split {
        let seps = find_separator_rows(&gray);
        if !seps.is_empty() {
        let (iw, ih) = (img.width(), img.height());
        let mut bounds: Vec<u32> = vec![0];
        bounds.extend(seps.iter().copied());
        bounds.push(ih);
        let exe = std::env::current_exe().expect("current exe");
        let mut pages = 0usize;
        for i in 0..bounds.len() - 1 {
            let y0 = bounds[i];
            let y1 = bounds[i + 1];
            if y1.saturating_sub(y0) < 50 {
                continue;
            }
            let page_img = img.crop_imm(0, y0, iw, y1 - y0);
            let page_path = out_dir.join(format!("page_{i}.png"));
            page_img.save(&page_path).expect("save page");
            println!("---- page_{i} ----");
            let child_out = out_dir.join(format!("page_{i}"));
            let mut cmd = std::process::Command::new(&exe);
            cmd.arg(&page_path).arg("--out").arg(&child_out).arg("--no-split");
            if dewarp {
                cmd.arg("--dewarp");
            } else {
                cmd.arg("--no-dewarp");
            }
            if gui {
                cmd.arg("--gui");
            }
            if keep_debug {
                cmd.arg("--debug");
            }
            let st = cmd.status().expect("run child");
            println!("page_{i}           : {}", if st.success() { "OK" } else { "FAILED" });
            pages += 1;
        }
                // ---- step 12: combined multi-page searchable PDF ----
        let mut pdf_pages: Vec<(u32, u32, Vec<u8>, Vec<(i32, i32, i32, i32, String)>)> = Vec::new();
        for i in 0..bounds.len() - 1 {
            let y0 = bounds[i];
            let y1 = bounds[i + 1];
            if y1.saturating_sub(y0) < 50 {
                continue;
            }
            let child_out = out_dir.join(format!("page_{i}"));
            let tsv_path = child_out.join("text_lines.tsv");
            if !tsv_path.exists() {
                continue;
            }
            let pimg_path = child_out.join("page.png");
            if !pimg_path.exists() {
                continue;
            }
            let pimg = image::open(&pimg_path).expect("open winner page").to_rgb8();
            let jpeg_tmp = child_out.join("tmp.jpg");
            pimg.save(&jpeg_tmp).expect("tmp jpg");
            let jpeg = fs::read(&jpeg_tmp).expect("read jpg");
            let _ = fs::remove_file(&jpeg_tmp);
            let mut words: Vec<(i32, i32, i32, i32, String)> = Vec::new();
            for line in fs::read_to_string(&tsv_path).unwrap_or_default().lines() {
                let f: Vec<&str> = line.split('\t').collect();
                if f.len() >= 5 {
                    if let (Ok(x), Ok(y), Ok(bw), Ok(bh)) = (
                        f[0].parse::<i32>(),
                        f[1].parse::<i32>(),
                        f[2].parse::<i32>(),
                        f[3].parse::<i32>(),
                    ) {
                        words.push((x, y, bw, bh, f[4].to_string()));
                    }
                }
            }
            pdf_pages.push((pimg.width(), pimg.height(), jpeg, words));
        }
        if !pdf_pages.is_empty() {
            let pdf = build_multi_pdf(&pdf_pages);
            let pdf_path = out_dir.join("scan_searchable.pdf");
            fs::write(&pdf_path, &pdf).expect("save combined pdf");
            println!("combined_pdf     : {} ({} pages)", pdf_path.display(), pdf_pages.len());
        }

        let mut all_text = String::new();
        for i in 0..bounds.len() - 1 {
            let tp = out_dir.join(format!("page_{i}")).join("text.txt");
            if let Ok(t) = fs::read_to_string(&tp) {
                all_text.push_str(&t);
                all_text.push_str("\n\n");
            }
        }
        if !all_text.is_empty() {
            fs::write(out_dir.join("text.txt"), all_text).ok();
        }
        if !keep_debug {
            for i in 0..bounds.len() - 1 {
                let _ = fs::remove_file(out_dir.join(format!("page_{i}.png")));
            }
        }

        println!("pages_found      : {pages}");
        println!("STEP11_OK");
        if open_out {
            let _ = std::process::Command::new("explorer").arg(&out_dir).spawn();
        }
        return;
        }
    }

    let (gw, gh) = (gray.width(), gray.height());
    let max_side = gw.max(gh);
    const DETECT_MAX: u32 = 2000;
    let scale = if max_side > DETECT_MAX { DETECT_MAX as f32 / max_side as f32 } else { 1.0 };
    let preview = if scale < 1.0 {
        imageops::resize(&gray, (gw as f32 * scale) as u32, (gh as f32 * scale) as u32, imageops::FilterType::Lanczos3)
    } else { gray.clone() };
    let (w, h) = (preview.width(), preview.height());

    let t = otsu_level(&preview);
    let mut dark_border = 0u64; let mut tot_border = 0u64;
    for (x, y, p) in preview.enumerate_pixels() {
        if x < 4 || y < 4 || x >= w - 4 || y >= h - 4 {
            tot_border += 1; if p.0[0] < t { dark_border += 1; }
        }
    }
    let doc_is_light = dark_border as f64 > 0.5 * tot_border as f64;
    let mut mask = GrayImage::new(w, h);
    for (x, y, p) in preview.enumerate_pixels() {
        let v = p.0[0]; let doc = if doc_is_light { v >= t } else { v < t };
        mask.put_pixel(x, y, Luma([if doc { 255 } else { 0 }]));
    }

    let labels = connected_components(&mask, Connectivity::Four, Luma([0u8]));
    let mut counts: Vec<u64> = Vec::new();
    for p in labels.pixels() {
        let l = p.0[0] as usize; if l >= counts.len() { counts.resize(l + 1, 0); } counts[l] += 1;
    }
    let mut best = 0u32; let mut best_count = 0u64;
    for (l, c) in counts.iter().enumerate() { if l > 0 && *c > best_count { best_count = *c; best = l as u32; } }
    let area_ratio = best_count as f64 / ((w as u64 * h as u64) as f64);

    let (mut min_x, mut min_y) = (w, h); let (mut max_x, mut max_y) = (0u32, 0u32);
    let mut area = 0u64;
    let mut min_s = u32::MAX; let mut max_s = 0u32;
    let mut min_d = i64::MAX; let mut max_d = i64::MIN;
    let (mut tl, mut tr, mut br, mut bl) = ((0, 0), (0, 0), (0, 0), (0, 0));

    for (x, y, p) in labels.enumerate_pixels() {
        if p.0[0] != best { continue; }
        area += 1;
        if x < min_x { min_x = x; } if x > max_x { max_x = x; }
        if y < min_y { min_y = y; } if y > max_y { max_y = y; }
        let s = x + y; let d = x as i64 - y as i64;
        if s < min_s { min_s = s; tl = (x, y); } if s > max_s { max_s = s; br = (x, y); }
        if d > max_d { max_d = d; tr = (x, y); } if d < min_d { min_d = d; bl = (x, y); }
    }

    let bbox_w = max_x.saturating_sub(min_x) + 1; let bbox_h = max_y.saturating_sub(min_y) + 1;
    let fill_ratio = if area > 0 { area as f64 / (bbox_w as f64 * bbox_h as f64) } else { 0.0 };
    let background_mode = area_ratio >= 0.2 && fill_ratio >= 0.5 && best > 0;

    let (mode, corners, scale_back) = if background_mode {
        ("background", (tl, tr, br, bl), 1.0 / scale)
    } else {
        ("full_frame", ((0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)), 1.0 / scale)
    };

    let src_corners = [
        (corners.0.0 as f32 * scale_back, corners.0.1 as f32 * scale_back),
        (corners.1.0 as f32 * scale_back, corners.1.1 as f32 * scale_back),
        (corners.2.0 as f32 * scale_back, corners.2.1 as f32 * scale_back),
        (corners.3.0 as f32 * scale_back, corners.3.1 as f32 * scale_back),
    ];

    let dst_w = dist(src_corners[0], src_corners[1]).max(dist(src_corners[2], src_corners[3])).round() as u32;
    let dst_h = dist(src_corners[0], src_corners[3]).max(dist(src_corners[1], src_corners[2])).round() as u32;
    let dst_corners = [ (0.0, 0.0), (dst_w as f32, 0.0), (dst_w as f32, dst_h as f32), (0.0, dst_h as f32) ];

    let cross_radius = ((w.max(h) / 30).max(10)) as i32;
    let cross_thickness = (cross_radius / 5).max(1);
    let rect_thickness = ((w.max(h) / 150).max(2)) as u32;

    let mut vis = RgbImage::from_fn(w, h, |x, y| {
        let v = preview.get_pixel(x, y).0[0];
        Rgb([v, v, v])
    });

    if background_mode {
        draw_thick_rect(&mut vis, Rgb([255u8, 0, 0]), min_x, min_y, bbox_w, bbox_h, rect_thickness);
    } else {
        draw_thick_rect(&mut vis, Rgb([255u8, 0, 0]), 0, 0, w, h, rect_thickness);
    }
    let text_blocks = draw_text_blocks(&mut vis, &preview, rect_thickness, w, h, &out_dir);
    for c in [corners.0, corners.1, corners.2, corners.3] {
        draw_thick_cross(&mut vis, Rgb([0u8, 255, 0]), c.0 as i32, c.1 as i32, cross_radius, cross_thickness);
    }

    let vis_path = out_dir.join("step5_detect.png");
    vis.save(&vis_path).expect("save debug image");

    let h = solve_homography(src_corners, dst_corners).expect("Homography solver failed");
    let warped = warp_perspective(&gray, &h, dst_w, dst_h);
    let warped_path = out_dir.join("step6_warped.png");
    warped.save(&warped_path).expect("save warped image");

    // ---- step 7: enhancement candidates ----
    let rgb = img.to_rgb8();
    let warped_rgb = warp_perspective_rgb(&rgb, &h, dst_w, dst_h);
    let (warped, warped_rgb) = if gui {
        let (ctrl, val, bow) = auto_landmarks(&warped);
        println!("gui_auto_bow     : {bow:.2}");
        let (mctrl, mval) = gui_session(&warped, &warped_rgb, &ctrl, &val);
        tps_warp(&warped, &warped_rgb, &mctrl, &mval, &out_dir)
    } else if dewarp {
        dewarp_stage(&warped, &warped_rgb, &out_dir)
    } else {
        (warped, warped_rgb)
    };
    warped_rgb.save(out_dir.join("step7_color_warped.png")).expect("save color warped");

    let magic = stretch_rgb(&warped_rgb);
    magic.save(out_dir.join("step7_magic_color.png")).expect("save magic color");

    let g_stretch = stretch_gray(&warped, 0.01, 0.99);
    g_stretch.save(out_dir.join("step7_gray_stretch.png")).expect("save gray stretch");

    let g_eq = equalize_histogram(&warped);
    g_eq.save(out_dir.join("step7_equalized.png")).expect("save equalized");

    let bw_s = sauvola(&warped, 15, 0.2);
    bw_s.save(out_dir.join("step7_bw_sauvola.png")).expect("save sauvola");

    let t2 = otsu_level(&warped);
    let mut bw_o = GrayImage::new(dst_w, dst_h);
    for (x, y, p) in warped.enumerate_pixels() {
        bw_o.put_pixel(x, y, Luma([if p.0[0] >= t2 { 255 } else { 0 }]));
    }
    bw_o.save(out_dir.join("step7_bw_otsu.png")).expect("save otsu bw");

    println!("enhancements     : color_warped, magic_color, gray_stretch, equalized, bw_sauvola, bw_otsu");
    println!("STEP7_OK");

    // ---- step 8: OCR all candidates, confidence picks the winner ----
    if let Some(tess) = find_tesseract() {
        let names = [
            "step7_color_warped.png",
            "step7_magic_color.png",
            "step7_gray_stretch.png",
            "step7_equalized.png",
            "step7_bw_sauvola.png",
            "step7_bw_otsu.png",
        ];
        let mut results: Vec<(&str, f64, usize)> = Vec::new();
        let mut max_words = 1usize;
        for name in names {
            let p = out_dir.join(name);
            let tsv = run_ocr(&tess, &p, true);
            let (conf, words) = ocr_stats(&tsv);
            results.push((name, conf, words));
            if words > max_words { max_words = words; }
        }

        let mut best_name = "";
        let mut best_score = -1f64;
        for r in results.iter() {
            let (name, conf, words) = (r.0, r.1, r.2);
            let score = (words as f64) * (conf / 100.0).sqrt();
            println!("ocr_candidate    : {name} words={words} conf={conf:.1} score={score:.1}");
            if score > best_score {
                best_score = score;
                best_name = name;
            }
        }
        println!("ocr_winner       : {best_name} score={best_score:.1}");

        let best_text = run_ocr(&tess, &out_dir.join(best_name), false);
        let final_text = if best_text.trim().is_empty() {
            let fallback = results.iter().max_by_key(|r| r.2).unwrap().0;
            println!("ocr_fallback     : {fallback} (winner text was empty)");
            run_ocr(&tess, &out_dir.join(fallback), false)
        } else {
            best_text
        };
        fs::write(out_dir.join("text.txt"), &final_text).expect("save ocr text");
        let _ = fs::copy(out_dir.join(best_name), out_dir.join("page.png"));

        // ---- step 10: searchable PDF ----
        let tsv_best = run_ocr(&tess, &out_dir.join(best_name), true);

        // group words into physical lines (block, par, line) for a clean text layer
        let mut line_map: Vec<(i32, i32, i32, Vec<(i32, i32, i32, i32, String, f64)>)> = Vec::new();
        for line in tsv_best.lines().skip(1) {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() >= 12 {
                let text = f[11].trim();
                if text.is_empty() { continue; }
                if let (Ok(blk), Ok(par), Ok(ln), Ok(x), Ok(y), Ok(bw2), Ok(bh2), Ok(cf)) = (
                    f[2].parse::<i32>(),
                    f[3].parse::<i32>(),
                    f[4].parse::<i32>(),
                    f[6].parse::<i32>(),
                    f[7].parse::<i32>(),
                    f[8].parse::<i32>(),
                    f[9].parse::<i32>(),
                    f[10].parse::<f64>(),
                ) {
                    if cf >= 0.0 {
                        let key = (blk, par, ln);
                        match line_map.iter_mut().find(|e| (e.0, e.1, e.2) == key) {
                            Some(e) => e.3.push((x, y, bw2, bh2, text.to_string(), cf)),
                            None => line_map.push((blk, par, ln, vec![(x, y, bw2, bh2, text.to_string(), cf)])),
                        }
                    }
                }
            }
        }

        let mut cropped = 0usize;
        let mut clean_text = String::new();
        let mut words: Vec<(i32, i32, i32, i32, String)> = Vec::new();
        if keep_debug { println!("detector_v6_conf"); }
        for (_, _, _, ws) in line_map.iter() {
            let mut ws = ws.clone();
            ws.sort_by_key(|w| w.0);
            let y = ws.iter().map(|w| w.1).min().unwrap();
            let bh = ws.iter().map(|w| w.3).max().unwrap();
            let avg_conf = ws.iter().map(|w| w.5).sum::<f64>() / ws.len() as f64;
            let sample: String = ws.iter().take(6).map(|w| w.4.as_str()).collect::<Vec<&str>>().join(" ");

            // Tesseract confidence is 0-100. Hallucinations/garbage average < 65.
            let is_garbage = avg_conf < 65.0;

            if keep_debug { println!("line_diag        : y={y} tbh={bh} conf={avg_conf:.1} flag={is_garbage} | {sample}"); }

            if is_garbage {
                cropped += 1;
                continue;
            }
            clean_text.push_str(&ws.iter().map(|w| w.4.as_str()).collect::<Vec<&str>>().join(" "));
            clean_text.push('\n');
            for w in ws.iter() {
                words.push((w.0, y, w.2, bh, w.4.clone()));
            }
        }
        println!("dropped_lines    : {cropped}");
        if !clean_text.is_empty() {
            fs::write(out_dir.join("text.txt"), &clean_text).expect("save clean text");
        }

        let mut tsv_out = String::new();
        for (x, y, bw, bh, text) in words.iter() {
            tsv_out.push_str(&format!("{x}\t{y}\t{bw}\t{bh}\t{text}\n"));
        }
        fs::write(out_dir.join("text_lines.tsv"), tsv_out).expect("save text lines");

        let tmp_jpg = out_dir.join("step10_tmp.jpg");
        warped_rgb.save(&tmp_jpg).expect("save tmp jpg");
        let jpeg_buf = fs::read(&tmp_jpg).expect("read tmp jpg");
        let _ = fs::remove_file(&tmp_jpg);

        let pdf = build_searchable_pdf(dst_w, dst_h, &jpeg_buf, &words);
        let pdf_path = out_dir.join("scan_searchable.pdf");
        fs::write(&pdf_path, &pdf).expect("save pdf");
        println!("pdf_lines        : {}", words.len());
        println!("pdf_saved_to     : {}", pdf_path.display());
        let json = serde_json::json!({
            "input": input.to_string_lossy(),
            "page_mode": mode,
            "corners": {
                "tl": [corners.0.0, corners.0.1],
                "tr": [corners.1.0, corners.1.1],
                "br": [corners.2.0, corners.2.1],
                "bl": [corners.3.0, corners.3.1]
            },
            "text_blocks": text_blocks.iter().map(|b| serde_json::json!([b.0, b.1, b.2, b.3])).collect::<Vec<_>>(),
            "candidates": results.iter().map(|r| serde_json::json!({"name": r.0, "words": r.2, "conf": r.1})).collect::<Vec<_>>(),
            "winner": best_name,
            "dropped_lines": cropped,
            "text": if clean_text.is_empty() { final_text.clone() } else { clean_text.clone() }
        });
        let json_path = out_dir.join("result.json");
        fs::write(&json_path, serde_json::to_string_pretty(&json).unwrap()).expect("save json");
        println!("json_saved_to    : {}", json_path.display());
        println!("STEP9_OK");
        println!("STEP10_OK");
        println!("STEP8_OK");
    } else {
        println!("TESSERACT_NOT_FOUND");
    }

    if !keep_debug {
        cleanup_debug(&out_dir);
    }
    if open_out {
        let _ = std::process::Command::new("explorer").arg(&out_dir).spawn();
    }

    println!("input            : {}", input.display());
    println!("page_mode        : {mode}");
    println!("debug_saved_to   : {}", vis_path.display());
    println!("warped_saved_to  : {}", warped_path.display());
    println!("STEP_VISUAL_OK");
}

















































