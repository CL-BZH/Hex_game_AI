#ifndef HEX_UI_H
#define HEX_UI_H

#include <algorithm>
#include <functional>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

#ifdef _NCURSES

#include <ncurses.h>

#include <cmath>

constexpr short int BLUE_PAIR{1};
constexpr short int RED_PAIR{2};
constexpr short int WHITE_PAIR{3};

constexpr char BLUE_CHAR{'B'};
constexpr char LOWERCASE_BLUE_CHAR{'b'};
constexpr char RED_CHAR{'R'};
constexpr char LOWERCASE_RED_CHAR{'r'};
constexpr char DOT_CHAR{'.'};
constexpr char DASH_CHAR{'-'};
constexpr char BACKSLASH_CHAR{'\\'};
constexpr char SLASH_CHAR{'/'};

inline void NEWLINE(int& x, int& y, int newline_x, WINDOW* w)
{
  x = newline_x;
  y += 1;
}

inline void B(int x, int y, WINDOW* w)
{
  wattron(w, COLOR_PAIR(BLUE_PAIR) | A_BOLD);
  mvwaddch(w, y, x, BLUE_CHAR);
  wattroff(w, COLOR_PAIR(BLUE_PAIR) | A_BOLD);
}

inline void b(int x, int y, WINDOW* w)
{
  wattron(w, COLOR_PAIR(BLUE_PAIR) | A_BOLD | A_BLINK);
  mvwaddch(w, y, x, LOWERCASE_BLUE_CHAR);
  wattroff(w, COLOR_PAIR(BLUE_PAIR) | A_BOLD | A_BLINK);
}

inline void R(int x, int y, WINDOW* w)
{
  wattron(w, COLOR_PAIR(RED_PAIR) | A_BOLD);
  mvwaddch(w, y, x, RED_CHAR);
  wattroff(w, COLOR_PAIR(RED_PAIR) | A_BOLD);
}

inline void r(int x, int y, WINDOW* w)
{
  wattron(w, COLOR_PAIR(RED_PAIR) | A_BOLD | A_BLINK);
  mvwaddch(w, y, x, LOWERCASE_RED_CHAR);
  wattroff(w, COLOR_PAIR(RED_PAIR) | A_BOLD | A_BLINK);
}

inline void DOT(int x, int y, WINDOW* w)
{
  wattron(w, COLOR_PAIR(WHITE_PAIR));
  mvwaddch(w, y, x, ACS_BULLET);
  wattroff(w, COLOR_PAIR(WHITE_PAIR));
}

inline void DASH(int x, int y, WINDOW* w, short int pair = WHITE_PAIR, unsigned int type = A_NORMAL)
{
  wattron(w, COLOR_PAIR(pair) | type);
  mvwaddch(w, y, x, DASH_CHAR);
  wattroff(w, COLOR_PAIR(pair) | type);
}

inline void BACKSLASH(int x, int y, WINDOW* w, short int pair = WHITE_PAIR,
                      unsigned int type = A_NORMAL)
{
  wattron(w, COLOR_PAIR(pair) | type);
  mvwaddch(w, y, x, BACKSLASH_CHAR);
  wattroff(w, COLOR_PAIR(pair) | type);
}

inline void SLASH(int x, int y, WINDOW* w, short int pair = WHITE_PAIR,
                  unsigned int type = A_NORMAL)
{
  wattron(w, COLOR_PAIR(pair) | type);
  mvwaddch(w, y, x, SLASH_CHAR);
  wattroff(w, COLOR_PAIR(pair) | type);
}

inline void DEFAULT(int x, int y, char c, WINDOW* w)
{
  wattron(w, COLOR_PAIR(WHITE_PAIR));
  mvwaddch(w, y, x, c);
  wattroff(w, COLOR_PAIR(WHITE_PAIR));
}

#endif

struct HexUI;

struct Point
{
  Point(int x = 0, int y = 0) : x{x}, y{y}
  {
  }
  int x;
  int y;

  // friend HexUI& operator<<(HexUI& out, const Point&);

  explicit operator std::string() const
  {
    std::stringstream sstr;
    sstr << "(" << x << ", " << y << ")";
    return sstr.str();
  }

  Point& operator=(const Point& rhs)
  {
    x = rhs.x;
    y = rhs.y;
    return *this;
  }

  // This is just for convinience (only use it to compare points
  // with same y value
  friend bool operator<(const Point& lhs, const Point& rhs)
  {
    // if(lhs.y == rhs.y)
    return lhs.x < rhs.x;
  }
};

enum class GraphEdgeType
{
  Dash,
  Slash,
  Backslash,
  Unknown
};

struct GraphEdge
{
  GraphEdge(const Point& p1, const Point& p2, GraphEdgeType et = GraphEdgeType::Unknown)
      : p1{p1}, p2{p2}, type{et}
  {
  }

  explicit operator std::string() const
  {
    std::stringstream sstr;
    char e;
    if (type == GraphEdgeType::Dash)
      e = '-';
    else if (type == GraphEdgeType::Slash)
      e = '/';
    else if (type == GraphEdgeType::Backslash)
      e = '\\';
    else
      throw std::runtime_error{"Unknown graph edge type"};

    sstr << "\n" << static_cast<std::string>(p1) << " " << e << static_cast<std::string>(p2);
    return sstr.str();
  }

  Point p1;
  Point p2;
  GraphEdgeType type;
};

bool compare_smaller_x(const Point& p1, const Point& p2)
{
  return p1.x < p2.x;
}
bool compare_greater_x(const Point& p1, const Point& p2)
{
  return p1.x > p2.x;
}

struct HexUI
{
  HexUI()
  {
  }

  void start()
  {
#ifdef _NCURSES
    initscr();
    curs_set(FALSE);

    cbreak();

    if (has_colors() == FALSE)
    {
      endwin();
      std::cout << "Your terminal does not support color" << std::endl;
      exit(1);
    }

    start_color();
    init_pair(BLUE_PAIR, COLOR_BLUE, COLOR_BLACK);
    init_pair(RED_PAIR, COLOR_RED, COLOR_BLACK);
    init_pair(WHITE_PAIR, COLOR_WHITE, COLOR_BLACK);

    // set up initial windows
    getmaxyx(stdscr, parent_y, parent_x);

    board = newwin(parent_y - inputw_size, parent_x - x_margin, 1, 0);
    inputw = newwin(inputw_size, parent_x - x_margin, 1, 0);

    wrefresh(stdscr);
    wrefresh(inputw);
    wrefresh(board);
    refresh();

#endif
  }

  template <typename T>
  HexUI& operator<<(T const& rhs)
  {
    sstr << rhs;
    print();
    return *this;
  }

  HexUI& operator<<(const Point& point)
  {
    std::stringstream sstr;
    sstr << "(" << point.x << ", " << point.y << ")";
    print(sstr.str());
    return *this;
  }

  template <typename T>
  HexUI& operator>>(T& rhs)
  {
#ifdef _NCURSES
    int num{0};
    int ch{wgetch(inputw)};
    int val{0};

    wrefresh(inputw);
    while (true)
    {
      if ((ch == '\n') || (ch == ' '))
      {
        rhs = val;
        return *this;
      }
      // Only implemented for integral input
      if (isdigit(ch))
      {
        val *= pow(6, num);
        val += (ch - 48);
        ++num;
      }
      ch = wgetch(inputw);
    }
#else
    std::cin >> rhs;
    return *this;
#endif
  }

  // Use this function to exit ncurses
  void wait_key()
  {
#ifdef _NCURSES
    wgetch(inputw);
    endwin();
#endif
  }

  // Print what is currently in sstr and empty it.
  void print()
  {
#ifdef _NCURSES
    wclear(inputw);
    mvwprintw(inputw, 1, 1, sstr.str().c_str());
    wrefresh(inputw);
    // This one create a problem
    // wgetch(inputw);
#else
    // Print to console
    std::cout << sstr.str();
#endif
    // Empty it
    sstr.str(std::string());
  }

  // TODO: add concept
  template <typename T>
  void print(T o)
  {
    sstr << o;
    print();
  }

  void draw_board(std::string str)
  {
#ifdef _NCURSES
    nc_draw(str);
#else
    print(str);
#endif
  }

private:
  std::stringstream sstr;

#ifdef _NCURSES

  int parent_x, parent_y;
  int inputw_size{10};
  int x_margin{0};

  WINDOW* board;
  WINDOW* inputw;

  std::vector<Point> points;

  void get_edges(std::vector<GraphEdge>& edges)
  {
    Point current_point;
    Point point;

    for (auto current_pt{std::begin(points)}; current_pt != std::end(points); ++current_pt)
    {
      for (auto pt{std::begin(points)}; pt != std::end(points); ++pt)
      {
        if ((current_pt->y == pt->y) && (current_pt->x == pt->x - 4))
        {
          GraphEdge graph_edge{*current_pt, *pt, GraphEdgeType::Dash};
          edges.push_back(graph_edge);
        }
        else if ((current_pt->y == pt->y) && (current_pt->x == pt->x + 4))
        {
          GraphEdge graph_edge{*pt, *current_pt, GraphEdgeType::Dash};
          edges.push_back(graph_edge);
        }
        else if ((current_pt->x + 2 == pt->x) && (current_pt->y - 2 == pt->y))
        {
          GraphEdge graph_edge{*current_pt, *pt, GraphEdgeType::Slash};
          edges.push_back(graph_edge);
        }
        else if ((current_pt->x - 2 == pt->x) && (current_pt->y + 2 == pt->y))
        {
          GraphEdge graph_edge{*pt, *current_pt, GraphEdgeType::Slash};
          edges.push_back(graph_edge);
        }
        else if ((current_pt->x - 2 == pt->x) && (current_pt->y - 2 == pt->y))
        {
          GraphEdge graph_edge{*current_pt, *pt, GraphEdgeType::Backslash};
          edges.push_back(graph_edge);
        }
        else if ((current_pt->x + 2 == pt->x) && (current_pt->y + 2 == pt->y))
        {
          GraphEdge graph_edge{*pt, *current_pt, GraphEdgeType::Backslash};
          edges.push_back(graph_edge);
        }
      }
    }
  }

  void draw_edges(const std::vector<GraphEdge>& edges, short int winner_color)
  {
    // std::stringstream sstr;

    for (auto edge : edges)
    {
      int nx{edge.p1.x};
      int ny{edge.p1.y};
      // sstr << static_cast<std::string>(edge);

      if (edge.type == GraphEdgeType::Dash)
      {
        nx += 2;
        DASH(nx, ny, board, winner_color, A_BLINK);
      }
      else if (edge.type == GraphEdgeType::Slash)
      {
        nx += 1;
        ny -= 1;
        SLASH(nx, ny, board, winner_color, A_BLINK);
      }
      else if (edge.type == GraphEdgeType::Backslash)
      {
        nx -= 1;
        ny -= 1;
        BACKSLASH(nx, ny, board, winner_color, A_BLINK);
      }
      else
        throw std::runtime_error{"Unknown graph edge type"};
    }

    // print(sstr.str());
    // wrefresh(board);
    // wgetch(inputw);
  }

  void show_winning_path(short int winner_color)
  {
    Point current_point;
    Point previous_point;

    std::vector<GraphEdge> edges;
    get_edges(edges);
    draw_edges(edges, winner_color);

    print("Press any key to exit");
    wrefresh(board);

    wgetch(inputw);
    endwin();
  }

  void nc_draw(const std::string& str)
  {
    wclear(board);
    wrefresh(stdscr);
    wrefresh(board);
    curs_set(0);

    int start_x{10}, start_y{10};
    int x{start_x}, y{start_y};
    short int winner_color{0};

    for (auto c : str)
    {
      switch (c)
      {
        case 'B':
        {
          B(x, y, board);
          break;
        }
        case 'b':
        {
          // Store the point
          points.push_back(Point(x, y));
          b(x, y, board);
          winner_color = BLUE_PAIR;
          break;
        }
        case 'R':
        {
          R(x, y, board);
          break;
        }
        case 'r':
        {
          // Store the point
          points.push_back(Point(x, y));
          r(x, y, board);
          winner_color = RED_PAIR;
          break;
        }
        case '-':
        {
          DASH(x, y, board);
          break;
        }
        case '.':
        {
          DOT(x, y, board);
          break;
        }
        case '\n':
        {
          NEWLINE(x, y, start_x, board);
          break;
        }
        case ' ':
        {
          break;
        }
        case '\\':
        {
          BACKSLASH(x, y, board);
          break;
        }
        case '/':
        {
          SLASH(x, y, board);
          break;
        }
        default:
        {
          DEFAULT(x, y, c, board);
          break;
        }
      }
      // Next x
      ++x;
    }

    wrefresh(board);

    // If there is a winning path (letters 'b' or 'r' then show the edges colored)
    if (winner_color) show_winning_path(winner_color);
  }

#endif
};

#endif  // HEX_UI_H
