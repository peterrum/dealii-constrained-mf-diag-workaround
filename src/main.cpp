#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools.h>

using namespace dealii;

const int dim    = 2;
const int degree = 1;

template <int dim, typename number>
class Wrapper
{
public:
  Wrapper(MatrixFree<dim, number> &  mf,
          MatrixFree<dim, number> &  mf_plain,
          AffineConstraints<double> &constraints)
    : mf(mf)
    , mf_plain(mf_plain)
    , constraints(constraints)
  {}

  void
  vmult(LinearAlgebra::distributed::Vector<number> &      dst,
        const LinearAlgebra::distributed::Vector<number> &src)
  {
    mf.cell_loop(&Wrapper::vmult_local, this, dst, src);
  }

  void
  vmult_local(const MatrixFree<dim, number> &                   data,
              LinearAlgebra::distributed::Vector<number> &      dst,
              const LinearAlgebra::distributed::Vector<number> &src,
              const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    FEEvaluation<dim, degree, degree + 1, 1, number> fe_eval(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        fe_eval.reinit(cell);
        fe_eval.read_dof_values(src);

        fe_eval.evaluate(false, true, false);
        for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
          fe_eval.submit_gradient(fe_eval.get_gradient(q), q);
        fe_eval.integrate(false, true);
        fe_eval.distribute_local_to_global(dst);
      }
  }

  void
  compute_diag(LinearAlgebra::distributed::Vector<number> &      dst,
               const LinearAlgebra::distributed::Vector<number> &src)
  {
    mf.cell_loop(&Wrapper::compute_diag_local, this, dst, src);
  }

  void
  compute_diag_local(const MatrixFree<dim, number> &             data,
                     LinearAlgebra::distributed::Vector<number> &dst,
                     const LinearAlgebra::distributed::Vector<number> & /*src*/,
                     const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    const bool is_mg = false;
    const bool is_dg = (data.get_dof_handler().get_fe().dofs_per_vertex == 0);

    FEEvaluation<dim, degree, degree + 1, 1, number> fe_eval(data);
    FEEvaluation<dim, degree, degree + 1, 1, number> fe_eval_plain(mf_plain);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        fe_eval.reinit(cell);
        fe_eval_plain.reinit(cell);

        // 1) initialize empty local diagonal
        AlignedVector<VectorizedArray<number>> diagonal(fe_eval.dofs_per_cell,
                                                        make_vectorized_array<number>(0.0));

        // 2) get DoF-indices
        std::vector<types::global_dof_index> dof_indices[VectorizedArray<number>::n_array_elements];
        for (unsigned int v = 0; v < data.n_components_filled(cell); v++)
          {
            dof_indices[v].resize(fe_eval.dofs_per_cell);

            auto cell_v = data.get_cell_iterator(cell, v);

            if (is_mg)
              cell_v->get_mg_dof_indices(dof_indices[v]);
            else
              cell_v->get_dof_indices(dof_indices[v]);

            if (!is_dg)
              {
                // in the case of CG: shape functions are not ordered
                // lexicographically see
                // (https://www.dealii.org/8.5.1/doxygen/deal.II/classFE__Q.html)
                // so we have to fix the order
                auto temp = dof_indices[v];
                for (unsigned int j = 0; j < dof_indices[v].size(); j++)
                  dof_indices[v][j] = temp[data.get_shape_info().lexicographic_numbering[j]];
              }
          }

        // 3) loop over all local DoFs and setup local diagonal entry by entry
        for (unsigned int i = 0; i < fe_eval.dofs_per_cell; ++i)
          {
            // 3a) zero out local source vector
            for (unsigned int j = 0; j < fe_eval.dofs_per_cell; ++j)
              fe_eval.begin_dof_values()[j] = make_vectorized_array<number>(0.0);

            // 3b) create standard basis taking care constraints
            for (unsigned int v = 0; v < data.n_components_filled(cell); v++)
                if (!constraints.is_constrained(dof_indices[v][i]))
                  {
                    fe_eval.begin_dof_values()[i][v] = 1.0;

                    for (unsigned int ii = 0; ii < dof_indices[v].size(); ii++)
                      {
                        if (!constraints.is_constrained(dof_indices[v][ii]))
                          continue;
                        auto &cs = *constraints.get_constraint_entries(dof_indices[v][ii]);
                        for (auto c : cs)
                            if (c.first == dof_indices[v][i])
                              fe_eval.begin_dof_values()[ii][v] = c.second;
                      }
                  }

            // 3c) perform stand matrix-free operation
            fe_eval.evaluate(false, true, false);
            for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
              fe_eval.submit_gradient(fe_eval.get_gradient(q), q);
            fe_eval.integrate(false, true);

            // 3d)
            VectorizedArray<number> temp = 0.0;
            for (unsigned int v = 0; v < data.n_components_filled(cell); v++)
                if (!constraints.is_constrained(dof_indices[v][i]))
                  {
                    temp = fe_eval.begin_dof_values()[i][v];

                    for (unsigned int ii = 0; ii < dof_indices[v].size(); ii++)
                      {
                        if (!constraints.is_constrained(dof_indices[v][ii]))
                          continue;
                        auto &cs = *constraints.get_constraint_entries(dof_indices[v][ii]);
                        for (auto c : cs)
                            if (c.first == dof_indices[v][i])
                              temp += fe_eval.begin_dof_values()[ii][v] * c.second;
                      }
                  }
            diagonal[i] = temp;
          }

        // 4) write local diagonal back to the global diagonal
        for (unsigned int i = 0; i < fe_eval_plain.dofs_per_cell; ++i)
          fe_eval_plain.begin_dof_values()[i] = diagonal[i];
        fe_eval_plain.distribute_local_to_global(dst);
      }
  }

  MatrixFree<dim, number> &  mf;
  MatrixFree<dim, number> &  mf_plain;
  AffineConstraints<double> &constraints;
};

int
main()
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);

  DoFHandler<dim> dof_handler(tria);
  FESystem<dim>   fe(FE_Q<dim>(degree), 1);
  dof_handler.distribute_dofs(fe);


  auto fu = [&](AffineConstraints<double> &constraints) {
    const QGauss<dim> quadrature_formula(degree + 1);

    FEValues<dim>                        fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients | update_quadrature_points | update_JxW_values);
    const unsigned int                   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int                   n_q_points    = quadrature_formula.size();
    FullMatrix<double>                   cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    SparsityPattern dsp(4, 4, 4);
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    dsp.compress();

    SparseMatrix<double> system_matrix(dsp);

    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                   endc = dof_handler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(local_dof_indices);
          cell_matrix = 0;
          fe_values.reinit(cell);

          for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) += (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                                      fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                                      fe_values.JxW(q_index));           // dx

          constraints.distribute_local_to_global(cell_matrix, local_dof_indices, system_matrix);
        }
    system_matrix.print(std::cout);
  };

  {
    AffineConstraints<double> constraints;
    fu(constraints);
  }

  {
    AffineConstraints<double> constraints;
    std::vector<bool>         lines = {false, true, true, true};
    constraints.add_lines(lines);
    constraints.add_entry(1, 0, 0.75);
    constraints.close();
    constraints.print(std::cout);
    fu(constraints);

    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
    additional_data.mapping_update_flags =
      (update_values | update_gradients | update_JxW_values | update_quadrature_points);
    MatrixFree<dim, double> mf;
    mf.reinit(dof_handler, constraints, QGauss<1>(degree + 1), additional_data);
    MatrixFree<dim, double>   mf_plain;
    AffineConstraints<double> constraints_empty;
    mf_plain.reinit(dof_handler, constraints_empty, QGauss<1>(degree + 1), additional_data);
    Wrapper<dim, double>                       w(mf, mf_plain, constraints);
    LinearAlgebra::distributed::Vector<double> src, dst, diag;
    mf.initialize_dof_vector(src);
    mf.initialize_dof_vector(dst);
    mf.initialize_dof_vector(diag);
    diag   = 0.0;
    src[0] = 1.0;

    w.vmult(dst, src);
    dst.print(std::cout);

    w.compute_diag(diag, src);
    diag.print(std::cout);
  }
}